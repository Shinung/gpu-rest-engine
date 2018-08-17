#ifndef DETECTION_H
#define DETECTION_H

#if defined(_WINDLL)
	#ifdef DETECTION_EXPORTS
	#define DETECTION_API __declspec(dllexport)
	#else
	#define DETECTION_API __declspec(dllimport)
	#endif
#else
	#define DETECTION_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

	typedef struct detection_ctx detection_ctx;

	DETECTION_API detection_ctx* detection_initialize(const char* model_file, const char* trained_file,
		const char* mean_file, const char* mean_value, const char* label_file);

	DETECTION_API const char* detection_inference(detection_ctx* ctx,
		char* buffer, size_t length);

	DETECTION_API void detection_destroy(detection_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif
