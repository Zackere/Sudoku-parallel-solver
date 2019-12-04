#include "../include/time_manager.hpp"

namespace timeManager {
namespace {
float total_ms_elapsed = 0;
} // namespace
void ResetTime() { total_ms_elapsed = 0; }
void AddTimeElapsed(float ms) { total_ms_elapsed += ms; }
float GetElapsedTime() { return total_ms_elapsed; }
} // namespace timeManager
