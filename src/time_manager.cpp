#include "../include/time_manager.hpp"

namespace timeManager {
namespace {
float cur_time = 0;
} // namespace
void ResetTime() { cur_time = 0; }
void AddTimeElapsed(float ms) { cur_time += ms; }
float GetElapsedTime() { return cur_time; }
} // namespace timeManager
