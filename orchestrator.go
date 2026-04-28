package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

type jobStatus string
type jobType string

const (
	statusQueued      jobStatus = "queued"
	statusRunning    jobStatus = "running"
	statusCompleted  jobStatus = "completed"
	statusFailed     jobStatus = "failed"
	workerTimeoutSecs          = 30
	artifactDir                = "./artifacts"
	datasetDir                 = "./datasets"
)

type jobConfig struct {
	Model        string         `json:"model"`
	Dataset      string         `json:"dataset"`
	Epochs       int            `json:"epochs"`
	BatchSize    int            `json:"batchSize"`
	LearningRate float64        `json:"learningRate,omitempty"`
	HiddenDims   []int          `json:"hiddenDims,omitempty"`
	Dropout      float64        `json:"dropout,omitempty"`
	Params       map[string]any `json:"params,omitempty"`
}

type windowStat struct {
	Window   int     `json:"window"`
	Accuracy float64 `json:"accuracy"`
	F1       float64 `json:"f1"`
	Loss     float64 `json:"loss"`
}

type jobMetrics struct {
	Accuracy          float64            `json:"accuracy,omitempty"`
	Loss              float64            `json:"loss,omitempty"`
	F1                float64            `json:"f1,omitempty"`
	Precision         float64            `json:"precision,omitempty"`
	Recall            float64            `json:"recall,omitempty"`
	RuntimeMs         int64              `json:"runtimeMs,omitempty"`
	LatencyMs         int64              `json:"latencyMs,omitempty"`
	Throughput        float64            `json:"throughput,omitempty"`
	EpochCurve        []float64          `json:"epochCurve,omitempty"`
	WindowStats       []windowStat       `json:"windowStats,omitempty"`
	ConfMatrix        [][]int            `json:"confMatrix,omitempty"`
	FeatureImportance map[string]float64 `json:"featureImportance,omitempty"`
}

type artifactRef struct {
	RunID     string    `json:"runId"`
	Type      string    `json:"type"`
	Path      string    `json:"path"`
	SizeBytes int64     `json:"sizeBytes"`
	CreatedAt time.Time `json:"createdAt"`
}

type job struct {
	RunID      string        `json:"runId"`
	Type       jobType       `json:"type"`
	Status     jobStatus     `json:"status"`
	Config     jobConfig     `json:"config"`
	Metrics    jobMetrics    `json:"metrics,omitempty"`
	Artifacts  []artifactRef `json:"artifacts,omitempty"`
	WorkerID   string        `json:"workerId,omitempty"`
	Error      string        `json:"error,omitempty"`
	CreatedAt  time.Time     `json:"createdAt"`
	UpdatedAt  time.Time     `json:"updatedAt"`
	StartedAt  *time.Time    `json:"startedAt,omitempty"`
	FinishedAt *time.Time    `json:"finishedAt,omitempty"`
}

type workerInfo struct {
	WorkerID    string    `json:"workerId"`
	LastSeen    time.Time `json:"lastSeen"`
	CurrentJob  string    `json:"currentJob,omitempty"`
	JobsHandled int       `json:"jobsHandled"`
	Status      string    `json:"status"`
}

type datasetInfo struct {
	Name         string             `json:"name"`
	Rows         int                `json:"rows"`
	Features     int                `json:"features"`
	Classes      int                `json:"classes"`
	ClassBalance map[string]float64 `json:"classBalance"`
	FeatureNames []string           `json:"featureNames"`
	Description  string             `json:"description"`
	LabelCol     string             `json:"labelCol,omitempty"`
	Source       string             `json:"source,omitempty"`
}

type datasetMeta struct {
	Name         string             `json:"name"`
	LabelCol     string             `json:"labelCol"`
	Description  string             `json:"description"`
	UploadedAt   string             `json:"uploadedAt"`
	Rows         int                `json:"rows"`
	Features     int                `json:"features"`
	Classes      int                `json:"classes"`
	ClassBalance map[string]float64 `json:"classBalance"`
	FeatureNames []string           `json:"featureNames"`
}

type submitRequest struct {
	Type   jobType   `json:"type"`
	Config jobConfig `json:"config"`
}

type updateRequest struct {
	Status    jobStatus     `json:"status"`
	Metrics   jobMetrics    `json:"metrics,omitempty"`
	Artifacts []artifactRef `json:"artifacts,omitempty"`
	WorkerID  string        `json:"workerId,omitempty"`
	Error     string        `json:"error,omitempty"`
}

type heartbeatRequest struct {
	WorkerID   string `json:"workerId"`
	CurrentJob string `json:"currentJob,omitempty"`
}

type store struct {
	mu      sync.RWMutex
	jobs    map[string]*job
	queue   []*job
	workers map[string]*workerInfo
}

func newStore() *store {
	return &store{jobs: make(map[string]*job), workers: make(map[string]*workerInfo)}
}

func (s *store) enqueue(j *job) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.jobs[j.RunID] = j
	s.queue = append(s.queue, j)
}

func (s *store) dequeue(workerID string) *job {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, j := range s.queue {
		if j.Status == statusQueued {
			s.queue = append(s.queue[:i], s.queue[i+1:]...)
			now := time.Now()
			j.Status = statusRunning
			j.WorkerID = workerID
			j.StartedAt = &now
			j.UpdatedAt = now
			if w, ok := s.workers[workerID]; ok {
				w.CurrentJob = j.RunID
				w.Status = "busy"
			}
			return j
		}
	}
	return nil
}

func (s *store) update(runID string, req updateRequest) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	j, ok := s.jobs[runID]
	if !ok {
		return fmt.Errorf("run %s not found", runID)
	}
	j.Status = req.Status
	j.Metrics = req.Metrics
	j.Error = req.Error
	if len(req.Artifacts) > 0 {
		j.Artifacts = append(j.Artifacts, req.Artifacts...)
	}
	now := time.Now()
	j.UpdatedAt = now
	if req.Status == statusCompleted || req.Status == statusFailed {
		j.FinishedAt = &now
		if w, ok := s.workers[req.WorkerID]; ok {
			w.CurrentJob = ""
			w.Status = "idle"
			w.JobsHandled++
		}
	}
	return nil
}

func (s *store) heartbeat(req heartbeatRequest) {
	s.mu.Lock()
	defer s.mu.Unlock()
	w, ok := s.workers[req.WorkerID]
	if !ok {
		w = &workerInfo{WorkerID: req.WorkerID, Status: "idle"}
		s.workers[req.WorkerID] = w
		log.Printf("worker registered: %s", req.WorkerID)
	}
	w.LastSeen = time.Now()
	if req.CurrentJob != "" {
		w.CurrentJob = req.CurrentJob
		w.Status = "busy"
	}
}

func (s *store) listWorkers() []*workerInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cutoff := time.Now().Add(-workerTimeoutSecs * time.Second)
	out := make([]*workerInfo, 0, len(s.workers))
	for _, w := range s.workers {
		wCopy := *w
		if wCopy.LastSeen.Before(cutoff) {
			wCopy.Status = "offline"
		}
		out = append(out, &wCopy)
	}
	return out
}

func (s *store) list(filterStatus, filterType string) []*job {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*job, 0, len(s.jobs))
	for _, j := range s.jobs {
		if filterStatus != "" && string(j.Status) != filterStatus {
			continue
		}
		if filterType != "" && string(j.Type) != filterType {
			continue
		}
		out = append(out, j)
	}
	sort.Slice(out, func(i, k int) bool {
		return out[i].CreatedAt.After(out[k].CreatedAt)
	})
	return out
}

func (s *store) get(runID string) (*job, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	j, ok := s.jobs[runID]
	return j, ok
}

func (s *store) stats() map[string]any {
	s.mu.RLock()
	defer s.mu.RUnlock()
	counts := map[string]int{"queued": 0, "running": 0, "completed": 0, "failed": 0}
	for _, j := range s.jobs {
		counts[string(j.Status)]++
	}
	return map[string]any{
		"totalRuns":    len(s.jobs),
		"byStatus":     counts,
		"totalWorkers": len(s.workers),
		"queueDepth":   len(s.queue),
	}
}

func builtInDatasetRegistry() map[string]datasetInfo {
	return map[string]datasetInfo{
		"sample_dataset": {
			Name: "sample_dataset", Rows: 500, Features: 10, Classes: 2,
			ClassBalance: map[string]float64{"0": 0.48, "1": 0.52},
			FeatureNames: []string{"feat_0", "feat_1", "feat_2", "feat_3", "feat_4", "feat_5", "feat_6", "feat_7", "feat_8", "feat_9"},
			Description:  "Synthetic binary classification. Label = sign(feat_0 + feat_1). Standard normal features.",
			LabelCol:     "label",
			Source:       "built_in",
		},
		"holdout_set": {
			Name: "holdout_set", Rows: 200, Features: 10, Classes: 2,
			ClassBalance: map[string]float64{"0": 0.50, "1": 0.50},
			FeatureNames: []string{"feat_0", "feat_1", "feat_2", "feat_3", "feat_4", "feat_5", "feat_6", "feat_7", "feat_8", "feat_9"},
			Description:  "Held-out eval split. Same distribution as sample_dataset, seed=99.",
			LabelCol:     "label",
			Source:       "built_in",
		},
		"time_series": {
			Name: "time_series", Rows: 600, Features: 8, Classes: 2,
			ClassBalance: map[string]float64{"0": 0.45, "1": 0.55},
			FeatureNames: []string{"lag_1", "lag_2", "lag_3", "rolling_mean", "rolling_std", "momentum", "rsi", "volume"},
			Description:  "Synthetic time-series for backtesting. Temporal ordering preserved; use rolling windows.",
			LabelCol:     "label",
			Source:       "built_in",
		},
		"imbalanced_set": {
			Name: "imbalanced_set", Rows: 400, Features: 10, Classes: 2,
			ClassBalance: map[string]float64{"0": 0.90, "1": 0.10},
			FeatureNames: []string{"feat_0", "feat_1", "feat_2", "feat_3", "feat_4", "feat_5", "feat_6", "feat_7", "feat_8", "feat_9"},
			Description:  "Imbalanced binary dataset (9:1 ratio). Prioritise F1/recall over accuracy.",
			LabelCol:     "label",
			Source:       "built_in",
		},
	}
}

func datasetRegistry() map[string]datasetInfo {
	reg := builtInDatasetRegistry()
	for name, info := range uploadedDatasetRegistry() {
		reg[name] = info
	}
	return reg
}

func uploadedDatasetRegistry() map[string]datasetInfo {
	out := make(map[string]datasetInfo)

	files, err := os.ReadDir(datasetDir)
	if err != nil {
		return out
	}

	for _, f := range files {
		if f.IsDir() || filepath.Ext(f.Name()) != ".csv" {
			continue
		}

		name := strings.TrimSuffix(f.Name(), ".csv")
		meta := readDatasetMeta(name)

		out[name] = datasetInfo{
			Name:         name,
			Rows:         meta.Rows,
			Features:     meta.Features,
			Classes:      meta.Classes,
			ClassBalance: meta.ClassBalance,
			FeatureNames: meta.FeatureNames,
			Description:  meta.Description,
			LabelCol:     meta.LabelCol,
			Source:       "uploaded",
		}
	}

	return out
}

func readDatasetMeta(name string) datasetMeta {
	meta := datasetMeta{
		Name:         name,
		LabelCol:     "label",
		Description:  "Uploaded CSV dataset.",
		ClassBalance: map[string]float64{},
		FeatureNames: []string{},
	}

	path := filepath.Join(datasetDir, name+".meta.json")
	b, err := os.ReadFile(path)
	if err != nil {
		return meta
	}

	json.Unmarshal(b, &meta)

	if meta.Name == "" {
		meta.Name = name
	}
	if meta.LabelCol == "" {
		meta.LabelCol = "label"
	}
	if meta.Description == "" {
		meta.Description = "Uploaded CSV dataset."
	}
	if meta.ClassBalance == nil {
		meta.ClassBalance = map[string]float64{}
	}
	if meta.FeatureNames == nil {
		meta.FeatureNames = []string{}
	}

	return meta
}

func inspectCsvDataset(path, name, labelCol string) (datasetInfo, error) {
	if labelCol == "" {
		labelCol = "label"
	}

	f, err := os.Open(path)
	if err != nil {
		return datasetInfo{}, err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1

	headers, err := reader.Read()
	if err != nil {
		return datasetInfo{}, err
	}

	labelIdx := -1
	featureNames := []string{}

	for i, h := range headers {
		h = strings.TrimSpace(h)
		headers[i] = h
		if h == labelCol {
			labelIdx = i
		}
	}

	if labelIdx == -1 {
		return datasetInfo{}, fmt.Errorf("label column %s not found", labelCol)
	}

	for i, h := range headers {
		if i != labelIdx {
			featureNames = append(featureNames, h)
		}
	}

	rows := 0
	classCounts := map[string]int{}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return datasetInfo{}, err
		}
		if len(record) <= labelIdx {
			continue
		}
		label := strings.TrimSpace(record[labelIdx])
		if label == "" {
			label = "missing"
		}
		classCounts[label]++
		rows++
	}

	classBalance := map[string]float64{}
	for k, v := range classCounts {
		if rows > 0 {
			classBalance[k] = math.Round((float64(v)/float64(rows))*10000) / 10000
		}
	}

	return datasetInfo{
		Name:         name,
		Rows:         rows,
		Features:     len(featureNames),
		Classes:      len(classCounts),
		ClassBalance: classBalance,
		FeatureNames: featureNames,
		Description:  "Uploaded CSV dataset.",
		LabelCol:     labelCol,
		Source:       "uploaded",
	}, nil
}

func sanitizeDatasetName(name string) string {
	name = strings.TrimSpace(strings.ToLower(name))
	name = strings.TrimSuffix(name, ".csv")

	re := regexp.MustCompile(`[^a-zA-Z0-9_\-]+`)
	name = re.ReplaceAllString(name, "_")
	name = strings.Trim(name, "_-")

	if name == "" {
		name = "uploaded_dataset"
	}

	return name
}

func validateCsvDataset(path, labelCol string) error {
	info, err := inspectCsvDataset(path, "tmp", labelCol)
	if err != nil {
		return err
	}
	if info.Rows == 0 {
		return fmt.Errorf("dataset has no rows")
	}
	if info.Features == 0 {
		return fmt.Errorf("dataset has no feature columns")
	}
	if info.Classes < 1 {
		return fmt.Errorf("dataset needs at least one label class")
	}
	return nil
}

func uploadDatasetHandler(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing CSV file"})
		return
	}
	defer file.Close()

	rawName := r.FormValue("name")
	if rawName == "" && header != nil {
		rawName = header.Filename
	}

	name := sanitizeDatasetName(rawName)
	labelCol := strings.TrimSpace(r.FormValue("labelCol"))
	if labelCol == "" {
		labelCol = "label"
	}

	description := strings.TrimSpace(r.FormValue("description"))
	if description == "" {
		description = "Uploaded CSV dataset."
	}

	if err := os.MkdirAll(datasetDir, 0755); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	tmpPath := filepath.Join(datasetDir, name+".tmp.csv")
	csvPath := filepath.Join(datasetDir, name+".csv")

	out, err := os.Create(tmpPath)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	if _, err := io.Copy(out, file); err != nil {
		out.Close()
		os.Remove(tmpPath)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	out.Close()

	if err := validateCsvDataset(tmpPath, labelCol); err != nil {
		os.Remove(tmpPath)
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}

	if err := os.Rename(tmpPath, csvPath); err != nil {
		os.Remove(tmpPath)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	info, err := inspectCsvDataset(csvPath, name, labelCol)
	if err != nil {
		writeJSON(w, http.StatusCreated, map[string]any{"name": name, "warning": err.Error()})
		return
	}

	info.Description = description

	meta := datasetMeta{
		Name:         name,
		LabelCol:     labelCol,
		Description:  description,
		UploadedAt:   time.Now().Format(time.RFC3339),
		Rows:         info.Rows,
		Features:     info.Features,
		Classes:      info.Classes,
		ClassBalance: info.ClassBalance,
		FeatureNames: info.FeatureNames,
	}

	metaBytes, _ := json.MarshalIndent(meta, "", "  ")
	os.WriteFile(filepath.Join(datasetDir, name+".meta.json"), metaBytes, 0644)

	writeJSON(w, http.StatusCreated, info)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)

	if code != http.StatusNoContent {
		json.NewEncoder(w).Encode(v)
	}
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func main() {
	os.MkdirAll(artifactDir, 0755)
	os.MkdirAll(datasetDir, 0755)

	s := newStore()
	mux := http.NewServeMux()

	mux.HandleFunc("POST /submit", func(w http.ResponseWriter, r *http.Request) {
		var req submitRequest

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}

		if req.Config.Epochs == 0 {
			req.Config.Epochs = 5
		}
		if req.Config.BatchSize == 0 {
			req.Config.BatchSize = 32
		}
		if req.Config.LearningRate == 0 {
			req.Config.LearningRate = 0.001
		}

		j := &job{
			RunID:     uuid.New().String(),
			Type:      req.Type,
			Status:    statusQueued,
			Config:    req.Config,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}

		s.enqueue(j)

		log.Printf("queued %s job %s (model=%s dataset=%s)", j.Type, j.RunID[:8], j.Config.Model, j.Config.Dataset)
		writeJSON(w, http.StatusCreated, j)
	})

	mux.HandleFunc("GET /next_job", func(w http.ResponseWriter, r *http.Request) {
		workerID := r.URL.Query().Get("workerId")
		if workerID == "" {
			workerID = "anonymous"
		}

		j := s.dequeue(workerID)
		if j == nil {
			writeJSON(w, http.StatusNoContent, nil)
			return
		}

		log.Printf("dispatched %s to worker %s", j.RunID[:8], workerID)
		writeJSON(w, http.StatusOK, j)
	})

	mux.HandleFunc("PUT /runs/{runId}", func(w http.ResponseWriter, r *http.Request) {
		runID := r.PathValue("runId")

		var req updateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}

		if err := s.update(runID, req); err != nil {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
			return
		}

		j, _ := s.get(runID)
		log.Printf("run %s → %s", runID[:8], req.Status)
		writeJSON(w, http.StatusOK, j)
	})

	mux.HandleFunc("POST /heartbeat", func(w http.ResponseWriter, r *http.Request) {
		var req heartbeatRequest

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}

		s.heartbeat(req)
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	mux.HandleFunc("GET /runs", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, s.list(r.URL.Query().Get("status"), r.URL.Query().Get("type")))
	})

	mux.HandleFunc("GET /runs/{runId}", func(w http.ResponseWriter, r *http.Request) {
		j, ok := s.get(r.PathValue("runId"))
		if !ok {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
			return
		}

		writeJSON(w, http.StatusOK, j)
	})

	mux.HandleFunc("GET /runs/{runId}/artifacts", func(w http.ResponseWriter, r *http.Request) {
		j, ok := s.get(r.PathValue("runId"))
		if !ok {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
			return
		}

		writeJSON(w, http.StatusOK, j.Artifacts)
	})

	mux.HandleFunc("GET /artifacts/{runId}/{filename}", func(w http.ResponseWriter, r *http.Request) {
		path := filepath.Join(artifactDir, r.PathValue("runId"), r.PathValue("filename"))

		if _, err := os.Stat(path); os.IsNotExist(err) {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "artifact not found"})
			return
		}

		http.ServeFile(w, r, path)
	})

	mux.HandleFunc("GET /workers", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, s.listWorkers())
	})

	mux.HandleFunc("POST /datasets/upload", uploadDatasetHandler)

	mux.HandleFunc("GET /datasets", func(w http.ResponseWriter, r *http.Request) {
		reg := datasetRegistry()
		out := make([]datasetInfo, 0, len(reg))

		for _, d := range reg {
			out = append(out, d)
		}

		sort.Slice(out, func(i, j int) bool {
			if out[i].Source == out[j].Source {
				return out[i].Name < out[j].Name
			}
			return out[i].Source > out[j].Source
		})

		writeJSON(w, http.StatusOK, out)
	})

	mux.HandleFunc("GET /datasets/{name}", func(w http.ResponseWriter, r *http.Request) {
		d, ok := datasetRegistry()[r.PathValue("name")]
		if !ok {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "dataset not found"})
			return
		}

		writeJSON(w, http.StatusOK, d)
	})

	mux.HandleFunc("GET /stats", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, s.stats())
	})

	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{
			"status": "ok",
			"time":   time.Now().Format(time.RFC3339),
		})
	})

	log.Printf("orchestrator listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", corsMiddleware(mux)))
}