import osfclient

# Connect to your OSF project
project = osfclient.OSF().project('2abup')

# List all storage providers and their attributes
for store in project.storages:
    print(store._attributes)  # Shows all available attributes for each storage provider