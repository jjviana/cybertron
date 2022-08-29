// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package numpy

import "fmt"

type DTypeClass struct{}

type DType struct {
	DType string
	Align bool
	Copy  bool
}

func (DTypeClass) Call(args ...any) (any, error) {
	if len(args) == 0 || len(args) > 3 {
		return nil, fmt.Errorf("DTypeClass: want 1 to 3 arguments, got %#v", args)
	}

	dt := new(DType)

	var ok bool
	dt.DType, ok = args[0].(string)
	if !ok {
		return nil, fmt.Errorf("DTypeClass: want 1st arg (dtype) string, got %#v", args[0])
	}

	if len(args) > 1 {
		i, ok := args[1].(int)
		if !ok {
			return nil, fmt.Errorf("DTypeClass: want 2nd arg (align) int, got %#v", args[1])
		}
		dt.Align = i != 0
	}

	if len(args) > 2 {
		i, ok := args[2].(int)
		if !ok {
			return nil, fmt.Errorf("DTypeClass: want 3rd arg (copy) int, got %#v", args[2])
		}
		dt.Copy = i != 0
	}

	return dt, nil
}
