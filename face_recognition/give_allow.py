#!/usr/bin/python3

import json


def allow():
    username = input('\n [INPUT] Enter username: ')

    with open('./config/users.json', 'r') as f:
        data = json.load(f)
    with open('./config/users.json', 'w') as f:
        file_data = data
        file_data['allowed_users'].append(username)
        json.dump(file_data, f, indent=4)
    print(f' [INFO] {username} username appended to allowed users\n')


if __name__ == '__main__':
    allow()
