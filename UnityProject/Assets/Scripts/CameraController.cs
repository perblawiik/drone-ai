using UnityEngine;

public class CameraController : MonoBehaviour
{
    public CameraInput cameraInput;
    public float movementSpeed;

    // Update is called once per frame
    void Update()
    {
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");

        Vector3 moveDirection = new Vector3(horizontal, 0.0f, vertical) * movementSpeed * Time.deltaTime;
        transform.Translate(moveDirection);

        if (Input.GetKey(cameraInput.upKey))
        {
            transform.Translate(Vector3.up * Time.deltaTime * movementSpeed);
        }

        if (Input.GetKey(cameraInput.downKey))
        {
            transform.Translate(Vector3.up * Time.deltaTime * -movementSpeed);
        }

        //if (Input.GetKey(cameraInput.forwardKey))
        //{
        //    transform.Translate(transform.forward * Time.deltaTime * movementSpeed);
        //}
        //if (Input.GetKey(cameraInput.backKey))
        //{
        //    transform.Translate(transform.forward * Time.deltaTime * -movementSpeed);
        //}
        //if (Input.GetKey(cameraInput.leftKey))
        //{
        //    transform.Translate(transform.right * Time.deltaTime * -movementSpeed);
        //}
        //if (Input.GetKey(cameraInput.rightKey))
        //{
        //    transform.Translate(transform.right * Time.deltaTime * movementSpeed);
        //}
    }
}
