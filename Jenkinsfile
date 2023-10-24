pipeline {
    agent any
    stages {
        stage('Build') {
            parallel {
                stage('Build') {
                    steps {
                        sh 'echo "building the repo"'
                    }
                }
            }
        }
        stage('Copy Files') {
            steps {
                echo "Copying app.py and templates folder to another directory"
                sh 'sudo rm -r /home/ubuntu/templates'
                sh 'sudo rm -r /home/ubuntu/app.py'
                sh 'sudo cp /var/lib/jenkins/workspace/Flask-Backend_master/app.py /home/ubuntu'  // Replace /path/to/destination/ with the actual destination path
                sh 'sudo cp -r /var/lib/jenkins/workspace/Flask-Backend_master/templates /home/ubuntu'  // Recursively copy the templates folder
            }
        }
        stage('Deploy') {
            steps {
                echo "deploying the application"
                sh 'sudo systemctl restart my-flask-app.service'
            }
        }
    }
    post {
        always {
            echo 'The pipeline completed'
            junit allowEmptyResults: true, testResults: '**/test_reports/*.xml'
        }
        success {
            echo "Flask Application Up and running!!"
        }
        failure {
            echo 'Build stage failed'
            error('Stopping early…')
        }
    }
}
