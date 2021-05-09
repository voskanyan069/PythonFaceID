#!/bin/bash

rm ./dataset/*
rm ./trainer/*
echo """{
    \"names\": [
        \"None\"
    ]
}""" > ./config/users.json
