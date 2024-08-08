#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <chrono>
#include <random>
#include <set>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "yolov9.h"
#include "bytetrack/BYTETracker.h"
#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/json.hpp>
#include <opencv2/opencv.hpp>

#include <iomanip> // For std::put_time
#include <sstream> // For std::stringstream

using namespace std;
using namespace cv;

static unordered_map<int, unordered_set<int>> currentSecondVehiclesByCategory;
static deque<unordered_map<int, unordered_set<int>>> recentSecondsVehiclesByCategory;
const int HISTORY_SECONDS = 60; // Number of seconds to keep history

static vector<Scalar> colors;
static mongocxx::instance inst{};
static mongocxx::client conn{mongocxx::uri{}};
static mongocxx::database db = conn["alarabia"];

// Function to get current timestamp in HH:MM:SS format
std::string getTimestamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_time_t);

    std::stringstream ss;
    ss << std::put_time(&tm, "%H:%M:%S");

    return ss.str();
}

std::string getVehicleType(int class_id)
{
    switch (class_id)
    {
    case 1:
        return "bicycle";
    case 2:
        return "car";
    case 3:
        return "motorcycle";
    case 5:
        return "bus";
    case 7:
        return "truck";
    default:
        return "unknown";
    }
}

// Function to map detection class ID to category ID
int getCategoryId(int class_id)
{
    // Assuming a simple mapping where class_id is the same as category_id
    return class_id;
}

void processVehicleIds(const vector<byte_track::BYTETracker::STrackPtr> &outputs)
{
    for (const auto &output : outputs)
    {
        int vehicleId = output->getTrackId();
        int categoryId = getCategoryId(output->getClassId()); // Assuming getClassId() method exists or you have a way to determine class ID
        currentSecondVehiclesByCategory[categoryId].insert(vehicleId);
    }
}

void endOfSecond()
{
    // Calculate timestamp for the current second
    std::string timestamp = getTimestamp();

    // Initialize counts for each vehicle type
    std::map<std::string, int> vehicle_counts_by_type = {
        {"bicycle", 0},
        {"car", 0},
        {"motorcycle", 0},
        {"bus", 0},
        {"truck", 0}};

    // Create a set to track vehicle IDs that have already been counted in previous seconds
    std::set<int> previously_counted_vehicle_ids;

    // Populate the set with vehicle IDs from previous seconds
    for (const auto &previousSecond : recentSecondsVehiclesByCategory)
    {
        for (const auto &[_, vehicleIds] : previousSecond)
        {
            previously_counted_vehicle_ids.insert(vehicleIds.begin(), vehicleIds.end());
        }
    }

    // Count vehicles by category, avoiding double-counting
    for (const auto &[category, vehicleIds] : currentSecondVehiclesByCategory)
    {
        std::string vehicle_type = getVehicleType(category);
        if (vehicle_counts_by_type.find(vehicle_type) != vehicle_counts_by_type.end())
        {
            for (const auto &vehicleId : vehicleIds)
            {
                // Only count this vehicle ID if it hasn't been counted before
                if (previously_counted_vehicle_ids.find(vehicleId) == previously_counted_vehicle_ids.end())
                {
                    vehicle_counts_by_type[vehicle_type]++;
                    // Mark this vehicle ID as counted for the next second's consideration
                    previously_counted_vehicle_ids.insert(vehicleId);
                }
            }
        }
    }

    // Create MongoDB document
    bsoncxx::builder::stream::document document{};
    document << "timestamp" << timestamp
             << "bicycle" << vehicle_counts_by_type["bicycle"]
             << "car" << vehicle_counts_by_type["car"]
             << "motorcycle" << vehicle_counts_by_type["motorcycle"]
             << "bus" << vehicle_counts_by_type["bus"]
             << "truck" << vehicle_counts_by_type["truck"];

    // Debug: Print the document before insertion
    cout << "MongoDB Document: " << bsoncxx::to_json(document.view()) << endl;

    // Insert the document into MongoDB
    try
    {
        auto collection = db["count_analytics"];
        auto result = collection.insert_one(document.view());
        if (result)
        {
            cout << "Document inserted with ID: " << result->inserted_id().get_oid().value.to_string() << endl;
        }
        else
        {
            cerr << "Failed to insert document" << endl;
        }
    }
    catch (const std::exception &e)
    {
        cerr << "Exception: " << e.what() << endl;
    }

    // Clear the vehicle IDs for the current second and add to the recent history
    recentSecondsVehiclesByCategory.push_back(currentSecondVehiclesByCategory);
    currentSecondVehiclesByCategory.clear();

    // Maintain the size of the deque
    if (recentSecondsVehiclesByCategory.size() > HISTORY_SECONDS)
    {
        recentSecondsVehiclesByCategory.pop_front();
    }
}

void format_tracker_input(Mat &frame, vector<Detection> &detections, vector<byte_track::Object> &tracker_objects)
{
    const float H = 640;
    const float W = 640;
    const float r_h = H / (float)frame.rows;
    const float r_w = W / (float)frame.cols;

    // Define a set of class IDs to include
    const set<int> vehicle_classes = {1, 2, 3, 5, 7};

    for (const auto &detection : detections)
    {
        // Check if the detection class ID is in the set of vehicle classes
        if (vehicle_classes.find(detection.class_id) != vehicle_classes.end())
        {
            float x = detection.bbox.x;
            float y = detection.bbox.y;
            float width = detection.bbox.width;
            float height = detection.bbox.height;

            if (r_h > r_w)
            {
                x = x / r_w;
                y = (y - (H - r_w * frame.rows) / 2) / r_w;
                width = width / r_w;
                height = height / r_w;
            }
            else
            {
                x = (x - (W - r_h * frame.cols) / 2) / r_h;
                y = y / r_h;
                width = width / r_h;
                height = height / r_h;
            }

            byte_track::Rect<float> rect(x, y, width, height);
            byte_track::Object obj(rect, detection.class_id, detection.conf);

            tracker_objects.push_back(obj);
        }
    }
}
void draw_bboxes(Mat &frame, const vector<byte_track::BYTETracker::STrackPtr> &output)
{
    for (const auto &detection : output)
    {
        auto trackId = detection->getTrackId();
        int categoryId = getCategoryId(detection->getClassId());; // Assuming this method exists or you have a way to determine category ID

        int x = detection->getRect().tlwh[0];
        int y = detection->getRect().tlwh[1];
        int width = detection->getRect().tlwh[2];
        int height = detection->getRect().tlwh[3];

        auto color_id = trackId % colors.size();
        rectangle(frame, Point(x, y), Point(x + width, y + height), colors[color_id], 3);

        // Detection box text
        string classString = to_string(trackId);
        Size textSize = getTextSize(classString, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect textBox(x, y - 40, textSize.width + 10, textSize.height + 20);
        rectangle(frame, textBox, colors[color_id], FILLED);
        putText(frame, classString, Point(x + 5, y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}

int main(int argc, char **argv)
{
    const string engine_path{argv[1]};
    const string video_path{argv[2]};
    assert(argc == 3);

    // Init model
    Yolov9 model(engine_path);

    // Init tracker
    byte_track::BYTETracker tracker(30, 30);

    // Store random colors
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(100, 255);
    for (int i = 0; i < 100; i++)
    {
        Scalar color = Scalar(dis(gen), dis(gen), dis(gen));
        colors.push_back(color);
    }

    // Open the video
    VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        cerr << "Error opening video file" << endl;
        return -1;
    }

    // Create a VideoWriter object to save the processed video
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter output_video("output_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(width, height));

    auto start = chrono::steady_clock::now();
    int frame_count = 0;

    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        vector<Detection> detections;
        model.predict(frame, detections);
        vector<byte_track::Object> tracker_objects;
        format_tracker_input(frame, detections, tracker_objects);

        auto output = tracker.update(tracker_objects);
        draw_bboxes(frame, output);

        // Process vehicle IDs
        processVehicleIds(output);

        // Write the processed frame to the output video
        output_video.write(frame);

        auto end = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::seconds>(end - start).count();
        if (elapsed >= 1)
        {
            endOfSecond();
            start = chrono::steady_clock::now();
        }

        frame_count++;
        cout << "Processed frame: " << frame_count << endl;

        // Display the frame
        imshow("Output", frame);
        if (waitKey(1) == 27)
            break; // Press 'ESC' to exit
    }

    // Clean up
    cap.release();
    output_video.release();
    destroyAllWindows();

    return 0;
}

//   1: bicycle
//   2: car
//   3: motorcycle
//   5: bus
//   7: truck

//   0: person
//   4: airplane
//   6: train