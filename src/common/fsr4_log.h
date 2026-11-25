// Small, shared diagnostics for the live bridge and cold shader builds.
// Callers choose a bounded cadence; this never waits for the GPU.
#pragma once
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

static inline void fsr4_logf(const char *format, ...) {
    char message[3072], line[3328];
    va_list args;
    va_start(args, format);
    vsnprintf(message, sizeof message, format, args);
    va_end(args);
    struct timeval timestamp;
    gettimeofday(&timestamp, NULL);
    snprintf(line, sizeof line, "[fsr4] t=%lld.%03d pid=%d %s\n",
             (long long)timestamp.tv_sec, (int)(timestamp.tv_usec / 1000),
             (int)getpid(), message);
    fputs(line, stderr);
    const char *path = getenv("FSR4_HOOK_LOG");
    if (!path || !path[0]) return;
    FILE *file = fopen(path, "a");
    if (!file) return;
    fputs(line, file);
    fclose(file);
}
