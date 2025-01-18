# Notes for AWS Deployment

Follow these steps to deploy the Docker container on AWS:

## Amazon ECR (Elastic Container Registry)

Push the Docker image to Amazon ECR:

```bash
aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com
docker tag stock-agent:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/stock-agent:latest
docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/stock-agent:latest
```

## Amazon ECS (Elastic Container Service)

Use Amazon ECS for container orchestration, auto-scaling, and high availability.

## Amazon EKS (Elastic Kubernetes Service)

Deploy the container on Amazon EKS for Kubernetes-based orchestration.

## Lambda with Docker (optional)

Use this Docker image in an AWS Lambda function for serverless deployments (max image size: 10 GB).