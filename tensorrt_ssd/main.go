package main

// #cgo pkg-config: opencv cudart-9.0
// #cgo LDFLAGS: -L/go/src/tensorrt-engine/build -lssdEngine -L/opt/TensorRT-4.0.0.3/targets/x86_64-linux-gnu/lib -lnvinfer -lnvinfer_plugin -lnvcaffe_parser -lcudnn -lprotobuf -lglog -lboost_system -lboost_thread
// #include <stdlib.h>
// #include "detection.h"
import "C"
import "unsafe"

import (
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
)

var ctx *C.detection_ctx

func detector(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "", http.StatusMethodNotAllowed)
		return
	}

	buffer, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	cstr, err := C.detection_inference(ctx, (*C.char)(unsafe.Pointer(&buffer[0])), C.size_t(len(buffer)))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer C.free(unsafe.Pointer(cstr))
	io.WriteString(w, C.GoString(cstr))
}

func main() {
	model := C.CString(os.Args[1])
	trained := C.CString(os.Args[2])
	label := C.CString(os.Args[3])

	log.Println("Initializing SSD model")
	var err error
	ctx, err = C.detection_initialize(model, trained, C.CString(""), C.CString("104,117,123"), label)
	if err != nil {
		log.Fatalln("could not initialize LPR model:", err)
		return
	}
	defer C.detection_destroy(ctx)

	log.Println("Adding REST endpoint /api/detection")
	http.HandleFunc("/api/detection", detector)
	log.Println("Starting server listening on :8000")
	log.Fatal(http.ListenAndServe(":8000", nil))
}
