#pragma once

#include "Rect.h"

namespace byte_track
{
struct Object
{
    Rect<float> rect;
    int class_id;
    float prob;

    Object(const Rect<float> &_rect,
           const int &_class_id,
           const float &_prob);
};
}