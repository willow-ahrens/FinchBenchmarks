#define TIME_MAX 10.0
#define TRIAL_MAX 10000

template <typename Setup, typename Test>
double benchmark(Setup setup, Test test) {
  auto time_total = std::chrono::high_resolution_clock::duration(0);
  auto time_min = std::chrono::high_resolution_clock::duration(0);
  int trial = 0;
  while (trial < TRIAL_MAX) {
    setup();
    auto tic = std::chrono::high_resolution_clock::now();
    test();
    auto toc = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
    trial++;
    if (trial == 1 || time < time_min) {
      time_min = time;
    }
    time_total += time;
    if (time_total.count() * 1e-9 > TIME_MAX) {
      break;
    }
  }
  return time_min.count() * 1e-9;
}