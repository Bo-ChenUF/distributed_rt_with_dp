import os

def check_exist(name: str, type: str):
    """
    Check if a file or directory exists.
    
    Inputs:
        - name: str. name of the file or directory.
        - type: str, enum. 'file' for a file and 'dir' for a directory
    
    Outputs:
        - boolean 
    """
    if type == "dir":
        if not os.path.exists(name):
            print(f"Directory {name} does not exist, creating it...")
            os.makedirs(name)
            print(f"Directory {name} is created!")
            
        
    if type == "file":
        if not os.path.isfile(name):
            raise ValueError("{name} does not exist!")