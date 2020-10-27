using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MouseLook : MonoBehaviour
{
    public float sensitivity;
    private Vector2 look;
    private GameObject flyingCamera;

    // Start is called before the first frame update
    void Start()
    {
        flyingCamera = transform.parent.gameObject;
    }

    // Update is called once per frame
    void Update()
    {
        float horizontal = Input.GetAxis("Mouse X");
        float vertical = Input.GetAxis("Mouse Y");

        Vector2 mousePos = new Vector2(horizontal, vertical);
        look += (sensitivity * mousePos);

        look.y = Mathf.Clamp(look.y, -80.0f, 80.0f);

        transform.localRotation = Quaternion.AngleAxis(-look.y, Vector3.right);
        flyingCamera.transform.localRotation = Quaternion.AngleAxis(look.x, flyingCamera.transform.up);
    }
}
