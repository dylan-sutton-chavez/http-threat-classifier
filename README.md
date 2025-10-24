## 1. Using the Model for OWASP Detection

After understanding how the **simple perceptron** works, may seem too basic for complex tasks. However, now with **giant models with billions of parameters**, these classic algorithms are **efficient solutions**. The foundation for this lies in a principle described by Cover _"Thus the probability of separability shows a pronounced threshold effect when the number of patterns is equal to twice the number of dimensions..." — (Cover, 1958, p. 331)_. This suggests that many complex problems can become **linearly separable**, and thus solvable by simple models, provided the data is represented in a sufficiently **high-dimensional vector space**.

### 1.1 Class Separability and Perceptron Convergence in High Dimensions

This theory is not just academic; **On October 5th, 2025**, I have an **independent research project** for threat detection, focusing on **OWASP standards**. The primary obstacle was the **lack of large, labeled datasets** for training. To overcome this, I developed a methodology: first, generating thousands of **realistic, synthetic security logs** using reproducible randomness, and then leveraging a **Large Language Model (LLM)** to label each log as either an **"attack" or "normal"**. This approach of **"black-box distillation"** proved highly effective, allowing the creation of a **200,000-entry dataset** for approximately **$30 USD**, achieving over **95% consistency** compared to manual labeling.

```json
{
    "metadata": {
        "timestamp": "string",
        "source": {
            "ip": "string",
            "port": 0
    }
    },
    "http": {
        "method": "string",
        "user_agent": "string",
        "query_params": "string",
        "endpoint": "string"
    },
    "metrics": {
        "payload_size": 0.0,
        "response_size": 0.0,
        "response_time_ms": 0
    }
}
```

The results of training a **simple perceptron** on this **high-dimensional** data are highly promising. Initial tests have achieved up to **93% efficiency** in **OWASP Top 10 attacks detection**, with an estimated **inference cost savings of over 99%** compared to commercial enterprise solutions like **AWS WAF**. I drafted it into a research paper, involves refining the vectorization methods and implementing advanced parameters, such as a **context-aware weight function** mathematically interpreted as `w(c) * x + b`, to further enhance detection capabilities.

### 1.2 LLM-Assisted Geometric Active Perceptron (L-GAP)

The epsilon margin ($\epsilon$) has been introduced within the step function's activation criteria. During model inference, this margin identifies uncertainty samples (or 'gray cases'), where the model cannot confidently classify the input. These samples are then processed under the L-GAP methodology. This involves forwarding the examples to a Large Language Model (LLM)—which serves as an assistant to provide high-confidence labels—and establishing a process of incremental learning based on the fine tuning module, allowing the model to adapt with every batch of newly labeled data.

To enhance data quality, the technique of isotropic transformation (or rounding) will be applied during every training and fine-tuning cycle: "In order to detect outliers, we use a linear transformation. Let $M = E[xx^T]$ where $x$ is a sample... Consider the transformed space $z = A^{-1}x$... and we will refer to the transformation as rounding" — (John D. Dunagan, 2002, p. 21). This means that once the data is transformed into the high-dimensional space, we use the resulting geometry to identify geometric remoteness ($\beta$-Outliers). In doing so, we filter the LLM examples and prioritize only the most anomalous ones for human review.

### 1.3 Context in a Simple Perceptron

The reason why separating web attacks is so complex is that it is impossible to know whether someone is an attacker or not when a complex pattern appears. This happens because an attacker might try to access the `/admin` endpoint to crack the form or perform a scan (having a history of several other endpoints like `/config`). But it could also be an administrator if the isolated `/admin` event is analyzed. Therefore, a unique vector will be cached for each IP where the "temperature" of each prediction will be dynamically adjusted based on the weights _w(c)_.

### 1.4 Payload Vectorization with Character N-gram Hashing

​To guarantee the Perceptron's high-speed inference for payloads (e.g., SQL injections like ' OR '1'='1'), the vectorization process must be optimized to operate in near o(1) time, independent of vocabulary size.

​This is achieved by using Character N-gram Hashing, a technique that doesn't need an external dictionary and converts the variable-length payload into a fixed-size, high-dimensional vector with a determinated generation hash (salt).
