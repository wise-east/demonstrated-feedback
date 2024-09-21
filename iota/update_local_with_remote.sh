# take argument of remote 

remote_dir=$1 # either ubuntu or ec2-user

# opposite of the following command

if [ $remote_dir == "ubuntu" ]; then
    rsync -avz -e "ssh -o StrictHostKeyChecking=no -i ~/.ssh/jcho2024-us-east-1.pem" ubuntu@ec2-52-70-93-208.compute-1.amazonaws.com:/home/ubuntu/project/GREASE-IOTA ~/project/

elif [ $remote_dir == "ec2-user" ]; then
    rsync -avz -e "ssh -o StrictHostKeyChecking=no -i ~/.ssh/jcho2024-us-east-1.pem" ec2-user@ec2-34-229-213-195.compute-1.amazonaws.com:/home/ec2-user/project/GREASE-IOTA ~/project/

fi