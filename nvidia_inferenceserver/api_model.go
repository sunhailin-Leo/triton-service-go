package nvidia_inferenceserver

type ModelIndexRequestHTTPObj struct {
	RepoName string `json:"repository_name"`
	Ready    bool   `json:"ready"`
}

type CudaMemoryRegisterBodyHTTPObj struct {
	RawHandle any    `json:"raw_handle"`
	DeviceID  int64  `json:"device_id"`
	ByteSize  uint64 `json:"byte_size"`
}

type SystemMemoryRegisterBodyHTTPObj struct {
	Key      string `json:"key"`
	Offset   uint64 `json:"offset"`
	ByteSize uint64 `json:"byte_size"`
}

type TraceSettingRequestHTTPObj struct {
	TraceSetting any `json:"trace_setting"`
}
