using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SoundManager : MonoBehaviour
{
    public bool soundEnabled = false;
    public static AudioClip droneSound;
    public float volume;

    private Rigidbody droneBody;
    static AudioSource source;
    
    // Start is called before the first frame update
    void Start()
    {
        droneSound = Resources.Load<AudioClip>("drone-default");

        source = GetComponent<AudioSource>();
        source.volume = volume;
        source.loop = true;
        source.clip = droneSound;

        droneBody = gameObject.GetComponentInParent<Rigidbody>();

        if (soundEnabled) source.Play();
    }

    private void FixedUpdate()
    {
        if (!soundEnabled) return;

        source.pitch = 1 + (droneBody.velocity.magnitude / 75f);
    }
}
