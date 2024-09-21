
rsync -avz -e "ssh -o StrictHostKeyChecking=no -i ~/.ssh/jcho2024-us-east-1.pem" ~/project/GREASE-IOTA ubuntu@ec2-52-70-93-208.compute-1.amazonaws.com:/home/ubuntu/project
rsync -avz -e "ssh -o StrictHostKeyChecking=no -i ~/.ssh/jcho2024-us-east-1.pem" ~/project/GREASE-IOTA ec2-user@ec2-34-229-213-195.compute-1.amazonaws.com:/home/ec2-user/project


