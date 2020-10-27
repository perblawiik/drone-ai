using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class DroneScript : Agent
{
    public GameObject startPoint;
    public GameObject checkpoint;
    public GameObject goalArea;
    public GameObject gate;

    public float verticalSpeed;
    public float horizontalSpeed;
    public float turningSpeed;

    private Rigidbody rb;

    // Proportional coefficient
    public float Kp = 15.0f;
    // Integral coefficient
    public float Ki = 15.0f;
    // Derivative coefficient
    public float Kd = 400.0f;

    private float errorSignal;
    private float integral;
    private float referenceSignal;

    private float heightInterpolation = 1.0f;

    private float previousGoalDistance;
    private float closestGoalDistance;
    private float stepPenalty;

    public bool isLanded = true;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        stepPenalty = -1.0f / MaxStep;
      
        ResetEnvironment();
    }

    private void ResetEnvironment()
    {
        int difficulty = (int)Academy.Instance.EnvironmentParameters.GetWithDefault("difficulty_level", 3.0f);
        AdjustDifficultyLevel(difficulty);

        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        rb.position = startPoint.transform.position;
        rb.rotation = startPoint.transform.rotation;

        referenceSignal = rb.position.y;
        errorSignal = 0.0f;
        integral = 0.0f;

        RandomizeCheckpointPosition();
        RandomizeGatePosition();

        previousGoalDistance = Vector3.Distance(goalArea.transform.position, transform.position);
        closestGoalDistance = Vector3.Distance(goalArea.transform.position, transform.position);
        
        isLanded = true;
    }

    private void AdjustDifficultyLevel(int level)
    {
        float offsetWeight = 0.0f;
        switch (level)
        {
            case 1:
                offsetWeight = 4.0f;
                break;
            case 2:
                offsetWeight = 6.0f;
                break;
            case 3:
                offsetWeight = 8.0f;
                break;
            default:
                break;
        }

        RandomizePlatformPositions(offsetWeight);
    }

    private void RandomizeGatePosition()
    {
        float weight = 7.52f;
        float offsetX = (2.0f * Random.value * weight) - weight;
        float offsetY = Random.value * weight;
        gate.transform.localPosition = new Vector3(offsetX, offsetY, -1.0f);
    }

    private void RandomizePlatformPositions(float offsetWeight)
    {
        GameObject startPlatform = startPoint.transform.parent.gameObject;
        GameObject endPlatform = goalArea.transform.parent.gameObject;

        // Generate a random number in the interval [-offset, offset]
        float startPlatformOffsetX = (2.0f * Random.value * offsetWeight) - offsetWeight;
        startPlatform.transform.localPosition = new Vector3(startPlatformOffsetX, -4.0f, -8.0f);

        float endPlatformOffsetX = (2.0f * Random.value * offsetWeight) - offsetWeight;
        endPlatform.transform.localPosition = new Vector3(endPlatformOffsetX, -2.0f, offsetWeight);
    }

    private void RandomizeCheckpointPosition()
    {
        // Randomize between two intervals: [0, PI/4] , [3 * PI/4, PI]
        float theta = (Random.value > 0.5f) ? Random.value * 0.785398f : Random.value * (0.785398f) + 2.356194f;
        // Randomize in the interval [PI/6, PI/2]
        float phi = Random.value * 1.047197f + 0.523598f;
        float radius = 9.0f;

        // Convert to cartesian coordinates
        float x = radius * Mathf.Sin(phi) * Mathf.Cos(theta);
        float z = radius * Mathf.Sin(phi) * Mathf.Sin(theta);
        float y = radius * Mathf.Cos(phi);

        checkpoint.SetActive(true);
        checkpoint.transform.localPosition = new Vector3(x, y, z);
    }


    public void MoveAgent(float[] vectorAction)
    {
        // Hover state cancels out gravity
        Vector3 totalForce = -Physics.gravity * rb.mass;

        if (isLanded == false)
        {
            // Left/Right/No Action
            int turnDirection = (int)vectorAction[0];
            if (turnDirection == 1)
            {   // Turn Left
                rb.AddTorque(-Vector3.up * turningSpeed);
            }
            else if (turnDirection == 2)
            {   // Turn Right
                rb.AddTorque(Vector3.up * turningSpeed);
            }
            else
            {   // Break
                //rb.AddTorque(-rb.angularVelocity * 0.2f);
                rb.AddTorque(-rb.angularVelocity * 0.5f);
            }

            // Forward/No Action
            int isMovingForward = (int)vectorAction[1];
            if (isMovingForward == 1)
            {
                totalForce += transform.forward * horizontalSpeed;
            }
            else
            {   // Break
                //totalForce -= new Vector3(rb.velocity.x * 1.5f, 0.0f, rb.velocity.z * 1.5f);
                totalForce -= new Vector3(rb.velocity.x * 2.0f, 0.0f, rb.velocity.z * 2.0f);
            }
        }
        
        // Rise/Decline/No Action
        int verticalDirection = (int)vectorAction[2];
        if (verticalDirection == 1)
        {
            //heightInterpolation = Mathf.Min(heightInterpolation + 0.02f, 1.0f);
            referenceSignal += verticalSpeed * heightInterpolation;
        }
        else if (verticalDirection == 2)
        {
            //heightInterpolation = Mathf.Min(heightInterpolation + 0.02f, 1.0f);
            referenceSignal -= verticalSpeed * heightInterpolation;
        }

        // Compute Height Control
        totalForce += new Vector3(0.0f, PIDController(), 0.0f);

        // Add total force to rigidbody
        rb.AddForce(totalForce);
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        MoveAgent(vectorAction);

        // Penalty given each step
        AddReward(stepPenalty);

        // Add more penalty if the drone is staying landed
        if (isLanded) AddReward(2.0f*stepPenalty);
    }

    public override void OnEpisodeBegin()
    {
        ResetEnvironment();
    }

    public override void Heuristic(float[] actionsOut)
    {
        System.Array.Clear(actionsOut, 0, actionsOut.Length);

        // Turn Left
        if (Input.GetKey(KeyCode.A))
        {
            actionsOut[0] = 1.0f;
        }

        // Turn Right
        if (Input.GetKey(KeyCode.D))
        {
            actionsOut[0] = 2.0f;
        }

        // Move Forward
        if (Input.GetKey(KeyCode.W))
        {
            actionsOut[1] = 1.0f;
        }

        // Rise
        if (Input.GetKey(KeyCode.Space))
        {
            actionsOut[2] = 1.0f;
        }

        // Decline
        if (Input.GetKey(KeyCode.LeftControl))
        {
            actionsOut[2] = 2.0f;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Vector pointing to the goal from the agent
        Vector3 goalDirection = goalArea.transform.position - transform.position;
        //sensor.AddObservation(goalDirection);
        float currentGoalDistance = goalDirection.magnitude;

        if (currentGoalDistance < previousGoalDistance)
        {
            AddReward(-stepPenalty);
        }
        previousGoalDistance = currentGoalDistance;
        sensor.AddObservation(currentGoalDistance);

        if (currentGoalDistance < closestGoalDistance)
        {
            closestGoalDistance = currentGoalDistance;
            AddReward(-2.0f*stepPenalty);
        }

        sensor.AddObservation(rb.velocity);
        sensor.AddObservation(rb.angularVelocity);

        float tiltFactor = 1.0f - Vector3.Dot(Vector3.up, transform.up);
        sensor.AddObservation(tiltFactor);

        sensor.AddObservation(checkpoint.activeSelf);
    }

    private float PIDController()
    {
        float previousError = errorSignal;

        // Present error
        errorSignal = referenceSignal - rb.position.y;

        // Past errors
        integral += (errorSignal * 0.5f) * Time.deltaTime;

        // Future errors (how the error changes)
        float derivative = errorSignal - previousError;

        // The integral part removes the steady state error and the derivative reduces the oscillation and improves stability
        return Kp * errorSignal + Ki * integral + Kd * derivative;
    }

    private void FixedUpdate()
    {
        float tiltFactor = 1.0f - Vector3.Dot(Vector3.up, transform.up);
        if (tiltFactor > 0.15)
        {
            SetReward(-1.0f);
            EndEpisode();
        }
    }

    void OnTriggerStay(Collider other)
    {
        if (other.gameObject.CompareTag(goalArea.tag))
        {
            // Add reward for finding the goal
            AddReward(0.5f);

            // Add max reward for finding the goal after collecting the checkpoint
            if (!checkpoint.activeSelf) AddReward(1.0f);

            EndEpisode();
        }
        if (other.gameObject.CompareTag("Checkpoint"))
        {
            AddReward(0.5f);
            other.gameObject.SetActive(false);
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("LandingArea"))
        {
            isLanded = true;
        }
    }

    void OnTriggerExit(Collider other)
    {
        if (other.gameObject.CompareTag("LandingArea"))
        {
            isLanded = false;
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("BoundingWall"))
        {
            SetReward(-1.0f);
            EndEpisode();
        }
    }
}