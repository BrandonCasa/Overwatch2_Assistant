#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <map>
#include <Windows.h>
#include <vector>

using namespace std::chrono;
using namespace std::this_thread;
using namespace std;

std::ofstream mouse_output_file;
std::ofstream keyboard_output_file;
std::atomic<bool> recording(true);

#pragma comment(lib, "User32.lib")

POINT previous_position;
steady_clock::time_point start_time;
vector<int> allowed_key_codes = {
    87,
    65,
    83,
    68,
    69,
    81,
    32,
    16,
    17,
    1,
    2,
};
map<int, bool> key_pressed_state;
int record_time = 10;

void RecordMouseSpeed()
{
  while (recording)
  {
    POINT current_position;
    GetCursorPos(&current_position);
    steady_clock::time_point current_time = steady_clock::now();
    auto ms_duration = duration_cast<chrono::milliseconds>(current_time - start_time);

    int dx = (current_position.x - previous_position.x);
    int dy = (current_position.y - previous_position.y);

    if (dx != 0 || dy != 0)
    {
      mouse_output_file << dx << "," << dy << "," << ms_duration.count() << endl;
      previous_position = current_position;
    }

    sleep_for(microseconds(50));
  }
}

void RecordKeyPresses()
{
  while (recording)
  {
    steady_clock::time_point current_time = steady_clock::now();
    auto ms_duration = duration_cast<milliseconds>(current_time - start_time);
    for (int key_code : allowed_key_codes)
    {
      short key_state = GetAsyncKeyState(key_code);
      if (key_state & 0x0001 && !key_pressed_state[key_code])
      {
        keyboard_output_file << key_code << ",Press," << ms_duration.count() << endl;
        key_pressed_state[key_code] = true;
      }
      if (key_state & 0x8000 && key_pressed_state[key_code])
      {
        keyboard_output_file << key_code << ",Release," << ms_duration.count() << endl;
        key_pressed_state[key_code] = false;
      }
    }
    sleep_for(microseconds(50));
  }
}

int main()
{
  bool mouse_key_start = false;
  bool mouse_move_start = false;
  while (!mouse_key_start || !mouse_move_start || !recording)
  {
    POINT current_position;
    GetCursorPos(&current_position);

    mouse_move_start = (current_position.x == 960 && current_position.y == 540)
    mouse_key_start = (GetAsyncKeyState(0x01) & 0x0001)
    recording = (mouse_key_start && mouse_move_start)

    sleep_for(microseconds(50));
  }
  mouse_output_file.open("mouse_data.csv");
  mouse_output_file << "dx,dy,Time(ms)" << endl;
  keyboard_output_file.open("keyboard_data.csv");
  keyboard_output_file << "KeyCode,Event,Time(ms)" << endl;

  start_time = steady_clock::now();
  GetCursorPos(&previous_position);

  for (int key_code : allowed_key_codes)
  {
    key_pressed_state[key_code] = false;
  }

  thread mouse_thread(RecordMouseSpeed);
  thread keyboard_thread(RecordKeyPresses);

  sleep_for(seconds(record_time));
  recording = false;

  mouse_thread.join();
  keyboard_thread.join();

  mouse_output_file.close();
  keyboard_output_file.close();
  return 0;
}