// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	//lint:ignore ST1001 allow dot import just to make the example more readable
	. "github.com/nlpodyssey/cybertron/examples"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/zeroshotclassifier"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
	LoadDotenv()

	modelsDir := HasEnvVar("CYBERTRON_MODELS_DIR")
	modelName := HasEnvVar("CYBERTRON_MODEL")
	possibleClasses := HasEnvVar("CYBERTRON_ZERO_SHOT_POSSIBLE_CLASSES")

	m, err := tasks.Load[zeroshotclassifier.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: modelName})
	if err != nil {
		log.Fatal().Err(err).Send()
	}
	defer tasks.Finalize(m)

	params := zeroshotclassifier.Parameters{
		CandidateLabels:    strings.Split(possibleClasses, ","),
		HypothesisTemplate: zeroshotclassifier.DefaultHypothesisTemplate,
		MultiLabel:         true,
	}

	fn := func(text string) error {
		start := time.Now()
		result, err := m.Classify(context.Background(), text, params)
		if err != nil {
			return err
		}
		fmt.Println(time.Since(start).Seconds())
		fmt.Println(result)
		return nil
	}

	err = ForEachInput(os.Stdin, fn)
	if err != nil {
		log.Fatal().Err(err)
	}
}
