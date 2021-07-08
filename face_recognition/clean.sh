#!/bin/bash

rm ./dataset/*
echo '' > ./trainer/trainer.yml
echo """{
    \"names\": [],
	\"allowed_users\": []
}""" > ./config/users.json
