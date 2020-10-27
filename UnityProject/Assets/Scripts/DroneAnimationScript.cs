using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DroneAnimationScript : MonoBehaviour
{
    public float speed = 1.0f;
    public GameObject blade1;

    // Update is called once per frame
    void FixedUpdate()
    {
        blade1.transform.localRotation = Quaternion.Euler(0.0f, blade1.transform.localRotation.eulerAngles.y + Time.deltaTime * speed, 0.0f);
    }
}
