#include "Object.h"

byte_track::Object::Object(const Rect<float> &_rect,
                           const int &_class_id,
                           const float &_prob) : rect(_rect), class_id(_class_id), prob(_prob)
{
}