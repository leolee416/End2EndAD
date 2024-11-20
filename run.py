import subprocess

if __name__ == '__main__':
    # subprocess.call(['python', 'train.py', '-d', '/home/yang/git/udacity_test/End-to-End-Learning-for-Self-Driving-Cars-master/data'])
    subprocess.call(['python', 'drive.py', 'models/checkpoint_epoch_straight_big_1st_10.pth'])