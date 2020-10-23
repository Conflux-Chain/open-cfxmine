#define FNV_PRIME 0x01000193

#define fnv(x, y) ((x)*FNV_PRIME ^ (y))