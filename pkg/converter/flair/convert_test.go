// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestConvert(t *testing.T) {
	err := Convert[float32]("/home/m/Downloads/flair/ner-english-ontonotes-fast/", true)
	require.NoError(t, err)
}
