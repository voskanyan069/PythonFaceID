#!/bin/bash

rm ./dataset/*
rm ./trainer/*
echo """{
    \"names\": [],
	\"allowed_users\": []
}""" > ./config/users.json
