#!/bin/bash

wget https://sato-data.s3.us-east-2.amazonaws.com/sato_tables.tar.gz
tar -zvxf sato_tables.tar.gz

wget https://sato-data.s3.us-east-2.amazonaws.com/viznet_tables.tar.gz
tar -zvxf viznet_tables.tar.gz

