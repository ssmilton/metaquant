package main

import (
    "encoding/json"
    "fmt"
    "os"
)

type ModelOutput struct {
    ModelID string        `json:"model_id"`
    RunID   string        `json:"run_id"`
    Signals []SignalEntry `json:"signals"`
}

type SignalEntry struct {
    Timestamp  string                 `json:"timestamp"`
    SecurityID int                    `json:"security_id"`
    SignalType string                 `json:"signal_type"`
    Strength   float64                `json:"strength"`
    Confidence float64                `json:"confidence"`
    Meta       map[string]interface{} `json:"meta"`
}

func main() {
    // For illustration only. A real model would consume stdin payload.
    payload := make(map[string]interface{})
    decoder := json.NewDecoder(os.Stdin)
    decoder.Decode(&payload)

    output := ModelOutput{
        ModelID: payload["model_id"].(string),
        RunID:   payload["run_id"].(string),
        Signals: []SignalEntry{},
    }

    data, _ := json.Marshal(output)
    fmt.Print(string(data))
}
