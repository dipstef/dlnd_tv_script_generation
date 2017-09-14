#!/usr/bin/env bash
floyd init dlnd_tv_script_generation
floyd run --mode jupyter --gpu --env tensorflow-1.0
