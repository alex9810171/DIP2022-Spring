document.getElementById('formFile').addEventListener("change", function () {
    document.getElementById('input_image').src = URL.createObjectURL(this.files[0]);
    document.getElementById('input_image').classList.remove('d-none');
});

document.getElementById('upload_button').addEventListener('click', function () {
    this.disabled = true;
    document.getElementById('upload_spinner').classList.remove('d-none');
    // upload blob image url to server
    let form = new FormData();
    let blob_url = document.getElementById('input_image').src;
    // get the image from object url and send to server
    axios.get(blob_url, { responseType: 'blob' }).then(response => {
        return new File([response.data], 'inputImage')
    }).then(file => {
        form.append('image', file);
        form.append('size', document.getElementById('formSize').value);
        axios.post('/api/preprocessing', form).then(function (response) {
            console.log('Upload Success');
            document.getElementById('output_image').src = response.data.output;
            document.getElementById('result_image').src = response.data.result;
            document.getElementById('linkText').innerText = response.data.result;
            document.getElementById('output-tab').disabled = false;
            document.getElementById('result-tab').disabled = false;

            document.getElementById('result-tab').click();
        }).catch(function (error) {
            alert('Upload Failed');
        }).finally(() => {
            this.disabled = false;
            document.getElementById('upload_spinner').classList.add('d-none');
        });
    })
})

document.getElementById('download').addEventListener('click', function () {
    let image_link = document.getElementById('result_image').src;
    axios.get(image_link, { responseType: 'blob' }).then(response => {
        let link = document.createElement('a');
        link.download = 'result.jpg';
        link.href = URL.createObjectURL(response.data);
        link.click();
    });
});

document.getElementById('next_step').addEventListener('click', function () {
    image_url = document.getElementById('result_image').src;
    window.location.href = '/enhancement?image_url=' + image_url;
});