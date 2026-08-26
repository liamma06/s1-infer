#include "infer/thread_pool.h"
#include <iostream>

/*
    Logic : for each worker
    - sleep until work is ready
    - run current fn using the range in ranges_[i] 
    - mark task as done 
    - wait until all tasks done(work_ready_ = false) before potentionally looping again 
*/

ThreadPool::ThreadPool(size_t num_threads) {
    /*
        https://www.geeksforgeeks.org/cpp/multithreading-in-cpp/
        cv : https://www.geeksforgeeks.org/cpp/cpp-multithreading-condition-variables/
        mutex: https://www.geeksforgeeks.org/cpp/std-mutex-in-cpp/ 
    */
    for (size_t i = 0; i < num_threads; ++i){
        workers_.emplace_back([this, i]{
            while (true) {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]{ return work_ready_ || stop_; });

                if (stop_) break;

                auto [start, end] = ranges_[i];
                auto fn = current_fn_;
                lock.unlock();

                fn(start, end);

                lock.lock();
                tasks_remaining_--;
                if (tasks_remaining_ == 0) {
                    work_ready_ = false;
                    cv_.notify_all();
                }else{
                    cv_.wait(lock, [this]{ return !work_ready_ || stop_; });
                }

            }

        });
    }
}

ThreadPool::~ThreadPool(){
    //stop all threads 
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();

    //make sure all threads finished 
    for (auto& worker : workers_){
        worker.join(); 
    }
}

void ThreadPool::parallel_for(size_t N, const std::function<void(size_t start, size_t end)>& fn){
    /*
        N - > total work (match 2048 QKV projection)
    */

    size_t chunk_size = N / workers_.size();
    ranges_.clear();

    for (size_t i = 0; i < workers_.size(); ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == workers_.size() - 1) ? N : start + chunk_size;
        ranges_.emplace_back(start, end);
    }

    {
        std::unique_lock<std::mutex> lock(mutex_);
        current_fn_ = fn;
        tasks_remaining_ = workers_.size();
        work_ready_ = true;
    }
    cv_.notify_all();

    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]{ return tasks_remaining_ == 0; });
}
