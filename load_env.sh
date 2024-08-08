#!/bin/bash
   set -e

   tr -d '\r' < /env_data/.env > /env_data/.env.unix
   source /env_data/.env.unix
   exec "$@"