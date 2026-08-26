#pragma once
#include <functional>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <utility>

class ThreadPool {
    public:
        ThreadPool(size_t num_threads);
        ~ThreadPool();

        void parallel_for(size_t N, const std::function<void(size_t start, size_t end)>& fn);
        
    private:
        std::vector<std::thread> workers_;

        std::mutex mutex_;
        std::condition_variable cv_;
        bool stop_ = false;

        //this calls job + each worker's chunk 
        std::function<void(size_t, size_t)> current_fn_; 
        std::vector<std::pair<size_t, size_t>> ranges_;

        size_t tasks_remaining_ = 0;
        bool work_ready_ = false;
};
