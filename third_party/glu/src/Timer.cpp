#include "Timer.h"

int
Timer::start()
{
    if (gettimeofday (&this->old_wc_time, 0) == -1
        || getrusage (RUSAGE_SELF, &this->old_us_time) == -1)
        return -1;
    else
        return 0;
}

int
Timer::elapsedWallclockTime (double &wc)
{
    if (gettimeofday (&this->new_wc_time, 0) == -1)
        return -1;
    wc = (this->new_wc_time.tv_sec - this->old_wc_time.tv_sec) * 1000.0
         + (this->new_wc_time.tv_usec - this->old_wc_time.tv_usec) / 1000.0;
    return 0;
}

int
Timer::elapsedUserTime (double &ut)
{
    if (getrusage (RUSAGE_SELF, &this->new_us_time) == -1)
        return -1;
    ut = (this->new_us_time.ru_utime.tv_sec - this->old_us_time.ru_utime.tv_sec) * 1000.0
	        + ((this->new_us_time.ru_utime.tv_usec
            - this->old_us_time.ru_utime.tv_usec) / 1000.0);
    return 0;
}

int
Timer::elapsedSystemTime (double &st)
{
    if (getrusage (RUSAGE_SELF, &this->new_us_time) == -1)
        return -1;
    st = (this->new_us_time.ru_stime.tv_sec - this->old_us_time.ru_stime.tv_sec) * 1000.0
	        + ((this->new_us_time.ru_stime.tv_usec
            - this->old_us_time.ru_stime.tv_usec) / 1000.0);
    return 0;
}

int
Timer::elapsedTime (double &wallclock, double &user_time, double &system_time)
{
    if (this->elapsedWallclockTime (wallclock) == -1)
        return -1;
    else
    {
        if (getrusage (RUSAGE_SELF, &this->new_us_time) == -1)
	        return -1;
        user_time = (this->new_us_time.ru_utime.tv_sec
		                - this->old_us_time.ru_utime.tv_sec) * 1000.0
				        + ((this->new_us_time.ru_utime.tv_usec
			            - this->old_us_time.ru_utime.tv_usec) / 1000.0);
        system_time = (this->new_us_time.ru_stime.tv_sec
		                - this->old_us_time.ru_stime.tv_sec) * 1000.0
				        + ((this->new_us_time.ru_stime.tv_usec
			            - this->old_us_time.ru_stime.tv_usec) / 1000.0);
        return 0;
    }
}