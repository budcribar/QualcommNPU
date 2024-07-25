#ifndef SINGLE_CORE_BARREL_HEADER_H_
#define SINGLE_CORE_BARREL_HEADER_H_

#include "pickle_header_tags.h"

// get the pointer to the CONTENTS record of the barrel
static inline void *htp_header_get_contents_ptr(const void *buf, size_t buflen)
{
    // Make sure it's a barrel
    if (buflen < 32 || htp_header_get_MAGIC(buf) != Hdr_MAGIC_MULTI) return 0;

    void *toc_payload_p = nullptr;
    int const toc_payload_len = htp_header_locate_field(buf, buflen, HdrTag_CONTENTS, &toc_payload_p);

    if (toc_payload_len < int(4 * sizeof(unsigned))) return 0;

    return toc_payload_p;
}

// make sure it's a single core barrel.
// the number of nsps should be equal to 1.
// blob id should be (1 << 16u).
static inline size_t htp_header_verify_single_core_barrel(const void *buf, size_t buflen)
{
    unsigned const *toc_payload_p = (unsigned const *)htp_header_get_contents_ptr(buf, buflen);
    if (!toc_payload_p) return -1;

    // toc_ptr[0] is the location of `number of nsps`
    // make sure it's 1
    if (toc_payload_p[0] != 1) return -1;

    // just make sure that the blob id is correct !
    // toc_ptr[2] is the blod id
    unsigned int const nsp_index = 0;
    unsigned int const blob_id = (nsp_index + 1) << 16u;
    if (toc_payload_p[2] != blob_id) return -1;

    return 0;
}

// determine the number of NSPs in a barrel
static inline size_t htp_header_get_num_blobs(const void *buf, size_t buflen)
{
    unsigned const *toc_payload_p = (unsigned const *)htp_header_get_contents_ptr(buf, buflen);
    if (!toc_payload_p) return -1;

    // toc_ptr[0] is the location of `number of nsps`
    return toc_payload_p[0];
}

// For a single core barrel, determine the blob0 offset and blob_size
static inline size_t htp_header_single_core_barrel_loc(const void *buf, size_t buflen, size_t &blob_size)
{
    unsigned const *toc_payload_p = (unsigned const *)htp_header_get_contents_ptr(buf, buflen);
    if (!toc_payload_p) return -1;

    // blob_offset and blob size in bytes
    size_t blob_offset = toc_payload_p[1] << 4;
    blob_size = toc_payload_p[3] << 4;

    return blob_offset;
}

#endif /* SINGLE_CORE_BARREL_HEADER_H_ */