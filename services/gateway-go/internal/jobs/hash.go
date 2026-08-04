package jobs

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
)

// Compute stable SHA256 hash of configuration map
func hashConfig(config map[string]any) (string, error) {
	if config == nil {
		config = map[string]any{}
	}
	keys := make([]string, 0, len(config))
	for k := range config {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	ordered := make([]any, 0, len(keys)*2)
	for _, k := range keys {
		ordered = append(ordered, k, config[k])
	}
	b, err := json.Marshal(ordered)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}
