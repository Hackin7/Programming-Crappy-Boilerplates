#!/bin/sh

iverilog -o /tmp/verilog_test -s test test.v && vvp /tmp/verilog_test
