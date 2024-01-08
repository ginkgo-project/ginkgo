// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_STREAM_HPP_
#define GKO_PUBLIC_CORE_BASE_STREAM_HPP_


#include <ginkgo/core/base/executor.hpp>


namespace gko {


/**
 * An RAII wrapper for a custom CUDA stream.
 * The stream will be created on construction and destroyed when the lifetime of
 * the wrapper ends.
 */
class cuda_stream {
public:
    /** Creates an empty stream wrapper, representing the default stream. */
    cuda_stream();

    /**
     * Creates a new custom CUDA stream on the given device.
     *
     * @param device_id  the device ID to create the stream on.
     */
    cuda_stream(int device_id);

    /** Destroys the custom CUDA stream, if it isn't empty. */
    ~cuda_stream();

    cuda_stream(const cuda_stream&) = delete;

    /** Move-constructs from an existing stream, which will be emptied. */
    cuda_stream(cuda_stream&&);

    cuda_stream& operator=(const cuda_stream&) = delete;

    /** Move-assigns from an existing stream, which will be emptied. */
    cuda_stream& operator=(cuda_stream&&) = delete;

    /**
     * Returns the native CUDA stream handle.
     * In an empty cuda_stream, this will return nullptr.
     */
    CUstream_st* get() const;

private:
    CUstream_st* stream_;

    int device_id_;
};


/**
 * An RAII wrapper for a custom HIP stream.
 * The stream will be created on construction and destroyed when the lifetime of
 * the wrapper ends.
 */
class hip_stream {
public:
    /** Creates an empty stream wrapper, representing the default stream. */
    hip_stream();

    /**
     * Creates a new custom HIP stream on the given device.
     *
     * @param device_id  the device ID to create the stream on.
     */
    hip_stream(int device_id);

    /** Destroys the custom HIP stream, if it isn't empty. */
    ~hip_stream();

    hip_stream(const hip_stream&) = delete;

    /** Move-constructs from an existing stream, which will be emptied. */
    hip_stream(hip_stream&&);

    hip_stream& operator=(const hip_stream&) = delete;

    /** Move-assigns from an existing stream, which will be emptied. */
    hip_stream& operator=(hip_stream&&) = delete;

    /**
     * Returns the native HIP stream handle.
     * In an empty hip_stream, this will return nullptr.
     */
    GKO_HIP_STREAM_STRUCT* get() const;

private:
    GKO_HIP_STREAM_STRUCT* stream_;

    int device_id_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_STREAM_HPP_
