```mermaid
    classDiagram
    class TyperApp {
        +run()
    }
    class Main {
        +main()
    }
    class QuantumNeuralNetwork {
        +predict(input)
        -feedforward(input)
    }

    TyperApp --> Main : invokes
    Main --> QuantumNeuralNetwork : calls \n predict()
    QuantumNeuralNetwork --> predict : defines
    predict --> feedforward : calls
    ```
