# cython: language_level=3

cimport cython

import numpy as np
cimport numpy as np

from libc.stdlib cimport rand
from cython.parallel import prange
from libc.math cimport sqrt, fmax, exp
from libc.math cimport pow as fpow

cdef extern from "limits.h":
    int INT_MAX

cdef roll():
    return rand() / float(INT_MAX)

"""
cdef extern from "<algorithm>" namespace "std":
    cdef int imax[int](int x, int y)
    cdef int imin[int](int x, int y)
"""

DTYPE = np.float

ctypedef np.float_t DTYPE_t
ctypedef np.uint8_t uint8

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def calculate_profit_potential_stoploss(
        DTYPE_t[::1] ask_history,
        DTYPE_t[::1] bid_history,
        DTYPE_t stop_loss,
        DTYPE_t commission
    ):
    
    cdef Py_ssize_t n = ask_history.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t ci
    cdef Py_ssize_t m
    cdef DTYPE_t p_long_
    cdef DTYPE_t p_short_
    cdef bint stop_long_hit
    cdef bint stop_short_hit
    cdef DTYPE_t a
    cdef DTYPE_t b
    cdef DTYPE_t a0
    cdef DTYPE_t b0
    cdef DTYPE_t pos_value
    
    p_long = np.zeros(n, dtype=DTYPE)
    p_short = np.zeros(n, dtype=DTYPE)
    
    cdef DTYPE_t[::1] p_long_view = p_long
    cdef DTYPE_t[::1] p_short_view = p_short
    
    # outer loop
    for i in prange(n, nogil=True, num_threads=8):
    
        stop_long_hit = False
        stop_short_hit = False
        p_long_ = 0.0
        p_short_ = 0.0
    
        # the price at the opening of the position
        a0 = ask_history[i]
        b0 = bid_history[i]
        
        m = n - i - 1
    
        # inner loop from current tick to end of week
        for j in range(m):
            
            ci = i + j
    
            # the current price
            a = ask_history[ci]
            b = bid_history[ci]
    
            if not stop_long_hit:
                pos_value = a - b0 - commission
                p_long_ = max(pos_value, p_long_)
                stop_long_hit = pos_value < - stop_loss
                
            if not stop_short_hit:
                pos_value = a0 - b - commission
                p_short_ = max(pos_value, p_short_)
                stop_short_hit = pos_value < - stop_loss
                
            if stop_long_hit and stop_short_hit:
                break
        
        # accumulate p's
        p_long_view[i] = p_long_
        p_short_view[i] = p_short_

    return p_long, p_short


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def calculate_profit_potential_stoploss_takeprofit(
        DTYPE_t[::1] ask_history,
        DTYPE_t[::1] bid_history,
        DTYPE_t stop_loss,
        DTYPE_t take_profit,
        DTYPE_t commission
    ):
    
    cdef Py_ssize_t n = ask_history.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t ci
    cdef Py_ssize_t m
    cdef DTYPE_t p_long_
    cdef DTYPE_t p_short_
    cdef bint stop_long_hit
    cdef bint stop_short_hit
    cdef DTYPE_t a
    cdef DTYPE_t b
    cdef DTYPE_t a0
    cdef DTYPE_t b0
    cdef DTYPE_t pos_value
    
    p_long = np.zeros(n, dtype=DTYPE)
    p_short = np.zeros(n, dtype=DTYPE)
    
    cdef DTYPE_t[::1] p_long_view = p_long
    cdef DTYPE_t[::1] p_short_view = p_short
    
    # outer loop
    for i in prange(n, nogil=True, num_threads=8):
    
        stop_long_hit = False
        stop_short_hit = False
        p_long_ = 0.0
        p_short_ = 0.0
    
        # the price at the opening of the position
        a0 = ask_history[i]
        b0 = bid_history[i]
        
        m = n - i - 1
    
        # inner loop from current tick to end of week
        for j in range(m):
            
            ci = i + j
    
            # the current price
            a = ask_history[ci]
            b = bid_history[ci]
    
            if not stop_long_hit:
                pos_value = a - b0 - commission
                p_long_ = max(pos_value, p_long_)
                stop_long_hit = (pos_value < - stop_loss 
                                 or pos_value > take_profit)
                
            if not stop_short_hit:
                pos_value = a0 - b - commission
                p_short_ = max(pos_value, p_short_)
                stop_short_hit = (pos_value < - stop_loss 
                                  or pos_value > take_profit)
                
            if stop_long_hit and stop_short_hit:
                break
        
        # accumulate p's
        if p_long_ >= take_profit:
            p_long_view[i] = p_long_
        if p_short_ >= take_profit:
            p_short_view[i] = p_short_

    return p_long, p_short    


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def regular_causal_conv(
        DTYPE_t[::1] signal,
        DTYPE_t[::1] kernel
    ):
    
    cdef Py_ssize_t n = signal.shape[0]
    cdef Py_ssize_t k = kernel.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t pad = 1
    cdef DTYPE_t accum
    
    result = np.zeros(n, dtype=DTYPE)
    
    cdef DTYPE_t[::1] result_view = result
    
    # outer loop over signal
    for i in range(n - k):
        
        accum = 0.0
        
        # inner loop over kernel elements
        for j in range(k):
            
            accum += kernel[k - j - pad] * signal[i + k - j]
        
        result_view[i + k] = accum
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def regular_sma(
        DTYPE_t[::1] signal,
        Py_ssize_t w
    ):
    """ w is the window size """
    
    cdef Py_ssize_t i
    cdef Py_ssize_t n = signal.size
    cdef DTYPE_t buffer = 0.0
    
    result = np.zeros(n)
    
    cdef DTYPE_t[::1] result_view = result
    
    for i in range(n):
        
        buffer += signal[i]
        
        if i >= w:
            
            buffer -= signal[i - w]
        
        result_view[i] = buffer / w

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def regular_ema(
        DTYPE_t[::1] signal,
        DTYPE_t N
    ):
    """ N is the number of days in EMA. """
    
    cdef DTYPE_t beta = 2 / (N + 1)
    cdef DTYPE_t one_minus_beta = 1.0 - beta
    cdef Py_ssize_t i
    cdef Py_ssize_t n = signal.size
    
    result = np.zeros(n)
    
    cdef DTYPE_t[::1] result_view = result
    
    result_view[0] = signal[0]
    
    for i in range(n - 1):
        result_view[i + 1] = signal[i + 1] * beta + result_view[i] * one_minus_beta
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def regular_bbands(
        DTYPE_t[::1] signal,
        DTYPE_t N,
        Py_ssize_t std_w
    ):
    """ returns (ema(X, N), std(X, std_w)) where X is a regular time series """
    
    cdef DTYPE_t beta = 2 / (N + 1)
    cdef DTYPE_t one_minus_beta = 1.0 - beta
    cdef DTYPE_t buffer = 0.0
    cdef Py_ssize_t n = signal.size
    cdef Py_ssize_t i
    
    std = np.zeros(n)
    mean = np.zeros(n)
    
    cdef DTYPE_t[::1] std_view = std
    cdef DTYPE_t[::1] mean_view = mean
    
    mean_view[0] = signal[0]
    
    for i in range(n - 1):
        
        # calculate mean
        mean_view[i + 1] = signal[i + 1] * beta + mean_view[i] * one_minus_beta
        
        # calculate std
        # TODO - implement subtract last, add next moving average calc
        buffer += (mean_view[i + 1] - signal[i + 1]) ** 2
        
        if i + 1 >= std_w:
            
            buffer -= (mean_view[i + 1 - std_w] - signal[i + 1 - std_w]) ** 2
            
        std_view[i + 1] = sqrt(buffer / std_w)
    
    return mean, std


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def irregular_sma_eq(
        DTYPE_t[::1] signal,
        DTYPE_t [::1] times,
        Py_ssize_t tau
    ):
    """
    https://pdfs.semanticscholar.org/882e/93570eae184ae737bf0344cb50a2925e353d.pdf
    
    left = 1; roll_sum = 0;
    
    for (right in 1:N(X)) {
    
        // Expand window on right end
        roll_sum = roll_sum + values[right];
        
        // Shrink window on left end
        while (times[left] <= times[right] - tau) {
            roll_sum = roll_sum - values[left];
            left = left + 1;
        }
    
        // Save SMA value for current time window
        out[right] = roll_sum / (right - left + 1);
        
    }
    
    
    """
    
    cdef Py_ssize_t n = signal.size
    cdef Py_ssize_t right
    cdef Py_ssize_t left = 0
    cdef DTYPE_t roll_sum = 0.0
    
    result = np.zeros(n)
    
    cdef DTYPE_t[::1] result_view = result
    
    for right in range(n):
        
        roll_sum += signal[right]
        
        while times[left] <= (times[right] - tau):
            
            roll_sum -= signal[left]
            left += 1
        
        result_view[right] = roll_sum / (right - left + 1)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def irregular_sma(
        DTYPE_t[::1] signal,
        DTYPE_t [::1] times,
        DTYPE_t tau
    ):
    """
    https://pdfs.semanticscholar.org/882e/93570eae184ae737bf0344cb50a2925e353d.pdf
    
    left = 1; roll_area = left_area = values[1] * tau; out[1] = values[1];
    
    for (right in 2:N(X)) {
    
        // Expand interval on right end
        roll_area = roll_area + values[right-1] * (times[right] - times[right-1]);
        
        // Remove truncated area on left end
        roll_area = roll_area - left_area;
        
        // Shrink interval on left end
        t_left_new = times[right] - tau;
        
        while (times[left] <= t_left_new) {
            roll_area = roll_area - values[left] * (times[left+1] - times[left]);
            left = left + 1;
        }
        
        // Add truncated area on left end
        left_area = values[max(1, left-1)] * (times[left] - t_left_new)
        roll_area = roll_area + left_area;
        
        // Save SMA value for current time window
        out[right] = roll_area / tau;
    }
    
    """
    
    cdef Py_ssize_t n = signal.size
    cdef Py_ssize_t right
    cdef Py_ssize_t left = 0
    cdef DTYPE_t roll_area = signal[0] * tau
    cdef DTYPE_t left_area = signal[0] * tau
    cdef DTYPE_t roll_sum = 0.0
    cdef DTYPE_t t_left_new
    
    result = np.zeros(n)
    
    cdef DTYPE_t[::1] result_view = result
    
    result_view[0] = signal[0]
    
    for right in np.arange(1, n):
        
        roll_area += signal[right - 1] * (times[right] - times[right - 1])
        
        roll_area -= left_area
        
        t_left_new = times[right] - tau
        
        while (times[left] <= t_left_new):
            
            roll_area -= signal[left] * (times[left + 1] - times[left])
            left += 1
        
        left_area = signal[max(1, left - 1)] * (times[left] - t_left_new)
        roll_area += left_area
        
        result_view[right] = roll_area / tau

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def irregular_ema(
        DTYPE_t[::1] signal,
        DTYPE_t[::1] times,
        DTYPE_t tau
    ):
    cdef Py_ssize_t n = signal.size
    cdef Py_ssize_t i
    cdef DTYPE_t w
    
    result = np.zeros(n)
    
    cdef DTYPE_t[::1] result_view = result
    
    result_view[0] = signal[0]
    
    for i in np.arange(1, n):
        
        w = exp( - (times[i] - times[i - 1]) / tau )
        result_view[i] = result_view[i - 1] * w + signal[i - 1] * (1 - w)
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def irregular_max(
        DTYPE_t[::1] signal,
        DTYPE_t[::1] times,
        DTYPE_t tau
    ):
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t max_pos = 0
    cdef Py_ssize_t max_pos_
    cdef Py_ssize_t i
    cdef Py_ssize_t k
    cdef Py_ssize_t n = signal.size
    
    result = np.zeros(n)
    
    cdef DTYPE_t[::1] result_view = result 
    
    for right in range(n):
        
        if signal[right] >= signal[max_pos]:
            max_pos = right
        
        while times[left] <= (times[right] - tau):
            left += 1
        
        if max_pos < left:
            max_pos_ = left
            for k in range(right - left):
                if signal[left + k] > signal[max_pos_]:
                    max_pos_ = left + k
            max_pos = max_pos_
        
        result_view[right] = signal[max_pos]
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def irregular_min(
        DTYPE_t[::1] signal,
        DTYPE_t[::1] times,
        DTYPE_t tau
    ):
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t min_pos = 0
    cdef Py_ssize_t min_pos_
    cdef Py_ssize_t i
    cdef Py_ssize_t k
    cdef Py_ssize_t n = signal.size
    
    result = np.zeros(n)
    
    cdef DTYPE_t[::1] result_view = result 
    
    for right in range(n):
        
        if signal[right] <= signal[min_pos]:
            min_pos = right
        
        while times[left] <= (times[right] - tau):
            left += 1
        
        if min_pos < left:
            min_pos_ = left
            for k in range(right - left):
                if signal[left + k] < signal[min_pos_]:
                    min_pos_ = left + k
            min_pos = min_pos_
        
        result_view[right] = signal[min_pos]
    
    return result
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def irregular_minmax(
        DTYPE_t[::1] signal,
        DTYPE_t[::1] times,
        DTYPE_t tau
    ):
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t min_pos = 0
    cdef Py_ssize_t min_pos_
    cdef Py_ssize_t max_pos = 0
    cdef Py_ssize_t max_pos_
    cdef Py_ssize_t i
    cdef Py_ssize_t k
    cdef Py_ssize_t n = signal.size
    
    mins = np.zeros(n)
    maxes = np.zeros(n)
    
    cdef DTYPE_t[::1] mins_view = mins 
    cdef DTYPE_t[::1] maxes_view = maxes 
    
    for right in range(n):
    
        if signal[right] >= signal[max_pos]:
            max_pos = right
        
        if signal[right] <= signal[min_pos]:
            min_pos = right
        
        while times[left] <= (times[right] - tau):
            left += 1
            
        if max_pos < left:
            max_pos_ = left
            for k in range(right - left):
                if signal[left + k] > signal[max_pos_]:
                    max_pos_ = left + k
            max_pos = max_pos_
        
        if min_pos < left:
            min_pos_ = left
            for k in range(right - left):
                if signal[left + k] < signal[min_pos_]:
                    min_pos_ = left + k
            min_pos = min_pos_
        
        mins_view[right] = signal[min_pos]
        maxes_view[right] = signal[max_pos]
    
    return mins, maxes


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def irregular_roc(
        DTYPE_t[::1] signal,
        DTYPE_t[::1] times
    ):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = signal.size
    
    result = np.zeros(n)
    
    cdef DTYPE_t[::1] result_view = result
    
    result_view[0] = 0
    
    for i in np.arange(1, n):
        
        result_view[i] = (signal[i] - signal[i - 1]) / (times[i] - times[i - 1])
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def square_signal(
        DTYPE_t[::1] signal
    ):
    """
    https://pdfs.semanticscholar.org/882e/93570eae184ae737bf0344cb50a2925e353d.pdf
    
    sigma^2 = SMA(X^2, tau) - SMA(X, tau)^2
    
    """
    
    cdef Py_ssize_t n = signal.size
    cdef Py_ssize_t i
    
    result = np.zeros(n)
    
    cdef DTYPE_t[::1] result_view = result
    
    result_view[0] = signal[0]
    
    for i in range(n):
        
        result_view[i] = fpow(signal[i], 2.0)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def backtest_simple_bbands_strategy(
        DTYPE_t[::1] ask,
        DTYPE_t[::1] bid,
        DTYPE_t[::1] sma,
        DTYPE_t[::1] std,
        DTYPE_t bband_multiplier,
        DTYPE_t vrr,
        DTYPE_t pnl
    ):
    
    cdef bint long_open = False
    cdef bint short_open = False
    cdef DTYPE_t position_value = 0.0
    cdef DTYPE_t bband_higher
    cdef DTYPE_t bband_lower
    cdef DTYPE_t v_range
    cdef DTYPE_t take_profit
    cdef DTYPE_t stop_loss
    cdef Py_ssize_t i
    cdef Py_ssize_t n = ask.size
    cdef Py_ssize_t trade_counter = 0
    cdef Py_ssize_t opened_counter = 0
    cdef Py_ssize_t closed_counter = 0
    
    closed_value = np.zeros(np.round(len(ask) // 100))
    opened_value = np.zeros(np.round(len(ask) // 100))
    times_opened = np.zeros(np.round(len(ask) // 100))
    times_closed = np.zeros(np.round(len(ask) // 100))
    position_type = np.zeros(np.round(len(ask) // 100), dtype=np.uint8)
    
    cdef DTYPE_t[::1] closed_view = closed_value
    cdef DTYPE_t[::1] opened_view = opened_value
    cdef DTYPE_t[::1] times_opened_view = times_opened
    cdef DTYPE_t[::1] times_closed_view = times_closed
    cdef uint8[::1] position_type_view = position_type
    
    for i in range(n):
        
        v_range = std[i] * bband_multiplier
        bband_higher = sma[i] + v_range
        bband_lower = sma[i] - v_range
        
        if not (long_open or short_open):
            
            # short pos
            if ask[i] > bband_higher:
                
                short_open = True
                position_value = bid[i]
                opened_view[trade_counter] = position_value
                times_opened_view[trade_counter] = i
                position_type_view[trade_counter] = 2
                take_profit = v_range * vrr
                stop_loss = v_range * pnl * vrr
            
            # long pos
            if bid[i] < bband_lower:
                
                long_open = True
                position_value = ask[i]
                opened_view[trade_counter] = position_value
                times_opened_view[trade_counter] = i
                position_type_view[trade_counter] = 1
                take_profit = v_range * vrr
                stop_loss = v_range * pnl * vrr
        
        elif short_open:
            
            if (position_value - bid[i] >= take_profit
                or position_value - bid[i] <= - stop_loss):
                
                short_open = False
                closed_view[trade_counter] = position_value - bid[i]
                times_closed_view[trade_counter] = i
                trade_counter += 1
        
        elif long_open:
            
            if (ask[i] - position_value >= take_profit
                or ask[i] - position_value <= - stop_loss):
                
                long_open = False
                closed_view[trade_counter] = ask[i] - position_value
                times_closed_view[trade_counter] = i
                trade_counter += 1
        
    return position_type, times_opened, times_closed, opened_value, closed_value


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def backtest_turnback_bbands_strategy(
        DTYPE_t[::1] ask,
        DTYPE_t[::1] bid,
        DTYPE_t[::1] typ,
        DTYPE_t[::1] sma,
        DTYPE_t[::1] std,
        DTYPE_t bband_multiplier,
        DTYPE_t take_profit,
        DTYPE_t stop_loss
    ):
    """ typ is a smoothed typical price. it is tracked. if it goes outside the
    bband and comes back, a position is opened. """
    
    cdef bint long_open = False
    cdef bint short_open = False
    cdef bint short_pending = False
    cdef bint long_pending = False
    cdef DTYPE_t position_value = 0.0
    cdef DTYPE_t bband_higher
    cdef DTYPE_t bband_lower
    cdef DTYPE_t v_range
    cdef Py_ssize_t i
    cdef Py_ssize_t n = ask.size
    cdef Py_ssize_t trade_counter = 0
    cdef Py_ssize_t opened_counter = 0
    cdef Py_ssize_t closed_counter = 0
    
    closed_value = np.zeros(np.round(len(ask) // 100))
    opened_value = np.zeros(np.round(len(ask) // 100))
    times_opened = np.zeros(np.round(len(ask) // 100))
    times_closed = np.zeros(np.round(len(ask) // 100))
    position_type = np.zeros(np.round(len(ask) // 100), dtype=np.uint8)
    
    cdef DTYPE_t[::1] closed_view = closed_value
    cdef DTYPE_t[::1] opened_view = opened_value
    cdef DTYPE_t[::1] times_opened_view = times_opened
    cdef DTYPE_t[::1] times_closed_view = times_closed
    cdef uint8[::1] position_type_view = position_type
    
    for i in range(n):
        
        v_range = std[i] * bband_multiplier
        bband_higher = sma[i] + v_range
        bband_lower = sma[i] - v_range
        
        if not (long_open or short_open):
            
            # bband pierced upwards
            if (not short_pending and 
                typ[i] > bband_higher):
                short_pending = True
                
            # bband pierced downwards
            if (not long_pending and 
                typ[i] < bband_lower):
                long_pending = True
            
            # short pos
            if (short_pending and 
                typ[i] < bband_higher):
                
                short_open = True
                short_pending = False
                position_value = bid[i]
                opened_view[trade_counter] = position_value
                times_opened_view[trade_counter] = i
                position_type_view[trade_counter] = 2
            
            # long pos
            if (long_pending and 
                bid[i] > bband_lower):
                
                long_open = True
                long_pending = False
                position_value = ask[i]
                opened_view[trade_counter] = position_value
                times_opened_view[trade_counter] = i
                position_type_view[trade_counter] = 1
        
        elif short_open:
            
            if (position_value - bid[i] >= take_profit
                or position_value - bid[i] <= - stop_loss):
                
                short_open = False
                closed_view[trade_counter] = position_value - bid[i]
                times_closed_view[trade_counter] = i
                trade_counter += 1
        
        elif long_open:
            
            if (ask[i] - position_value >= take_profit
                or ask[i] - position_value <= - stop_loss):
                
                long_open = False
                closed_view[trade_counter] = ask[i] - position_value
                times_closed_view[trade_counter] = i
                trade_counter += 1
        
    return position_type, times_opened, times_closed, opened_value, closed_value
                

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def backtest_sma_crossover_strategy(
        DTYPE_t[::1] ask,
        DTYPE_t[::1] bid,
        DTYPE_t[::1] sma_slow,
        DTYPE_t[::1] sma_fast
    ):
    
    cdef bint regime_long = False
    cdef bint regime_long_new = False
    cdef DTYPE_t position_value = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t n = ask.size
    cdef Py_ssize_t trade_counter = 0
    
    closed_value = np.zeros(np.round(len(ask) / 100))
    opened_value = np.zeros(np.round(len(ask) / 100))
    times_opened = np.zeros(np.round(len(ask) // 100))
    times_closed = np.zeros(np.round(len(ask) // 100))
    position_type = np.zeros(np.round(len(ask) // 100), dtype=np.uint8)
    
    cdef DTYPE_t[::1] closed_view = closed_value
    cdef DTYPE_t[::1] opened_view = opened_value
    cdef DTYPE_t[::1] times_opened_view = times_opened
    cdef DTYPE_t[::1] times_closed_view = times_closed
    cdef uint8[::1] position_type_view = position_type
    
    regime_long = sma_fast[0] > sma_slow[0]
    
    for i in range(n):
        
        regime_long_new = sma_fast[i] < sma_slow[i]
        
        # crossover
        if regime_long_new != regime_long:
            
            # open long, close short
            if regime_long_new:
                
                # open new pos
                position_value_new = ask[i]
                opened_view[trade_counter] = position_value_new
                times_opened_view[trade_counter] = i
                position_type_view[trade_counter] = 1
                
                # close old short if exists
                if trade_counter > 0:
                    closed_view[trade_counter] = position_value - bid[i]
                    times_closed_view[trade_counter] = i
                    position_value = position_value_new
                    trade_counter += 1
            
            # open short, close long
            else:
                
                # open new pos
                position_value_new = bid[i]
                opened_view[trade_counter] = position_value_new
                times_opened_view[trade_counter] = i
                position_type_view[trade_counter] = 2
                
                # close old long if exists
                if trade_counter > 0:
                    closed_view[trade_counter] = ask[i] - position_value
                    times_closed_view[trade_counter] = i
                    position_value = position_value_new
                    trade_counter += 1
                
            regime_long = regime_long_new
    
    return position_type, times_opened, times_closed, opened_value, closed_value
