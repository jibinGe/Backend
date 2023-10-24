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
                sh 'cp app.py /home/ubuntu'  // Replace /path/to/destination/ with the actual destination path
                sh 'cp -r templates /home/ubuntu'  // Recursively copy the templates folder
            }
        }
        stage('Deploy') {
            steps {
                echo "deploying the application"
                sh "sudo systemctl restart my-flask-app.service"
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
