// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/flair"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/rs/zerolog/log"
)

const (
	// defaultPyModelFilename is the default Flair PyTorch model filename.
	defaultPyModelFilename = "pytorch_model.bin"
	// defaultGoModelFilename is the default Flair spaGO model filename.
	defaultGoModelFilename = "spago_model.bin"
)

// Convert converts a Flair PyTorch model to a Spago (Cybertron) model.
func Convert[T float.DType](modelDir string, overwriteIfExist bool) error {
	pyModelFilename := filepath.Join(modelDir, defaultPyModelFilename)
	goModelFilename := filepath.Join(modelDir, defaultGoModelFilename)

	if info, err := os.Stat(goModelFilename); !overwriteIfExist && err == nil && !info.IsDir() {
		log.Info().Str("model", goModelFilename).Msg("model file already exists, skipping conversion")
		return nil
	}

	st, err := flair.LoadSequenceTagger(pyModelFilename)
	if err != nil {
		return err
	}
	fmt.Println("SequenceTagger =", st)

	return nil
}
