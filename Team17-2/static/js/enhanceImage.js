var revert_images = [];

document.getElementById('formFile').addEventListener("change", function () {
    document.getElementById('input_image').src = URL.createObjectURL(this.files[0]);
    document.getElementById('result_image').src = document.getElementById('input_image').src;
    revert_images = [];
    document.getElementById('revert').disabled = true;
});

document.getElementById('tophat_button').addEventListener('click', function () {
    this.disabled = true;
    document.getElementById('tophat_spinner').classList.remove('d-none');
    // upload blob image url to server
    let form = new FormData(document.getElementById('tophat_options'));
    let blob_url = document.getElementById('result_image').src;
    // get the image from object url and send to server
    axios.get(blob_url, { responseType: 'blob' }).then(response => {
        return new File([response.data], 'inputImage')
    }).then(file => {
        form.append('image', file);
        form.append('method', 'tophat');
        axios.post('/api/enhancement', form).then(function (response) {
            console.log('Upload Success');
            revert_images.push(document.getElementById('result_image').src);
            document.getElementById('revert').disabled = false;
            document.getElementById('result_image').src = response.data.result;
        }).catch(function (error) {
            alert('Upload Failed');
        }).finally(() => {
            this.disabled = false;
            document.getElementById('tophat_spinner').classList.add('d-none');
        });
    })
})
document.getElementById('contrast_button').addEventListener('click', function () {
    this.disabled = true;
    document.getElementById('contrast_spinner').classList.remove('d-none');
    // upload blob image url to server
    let form = new FormData(document.getElementById('contrast_options'));
    let blob_url = document.getElementById('result_image').src;
    // get the image from object url and send to server
    axios.get(blob_url, { responseType: 'blob' }).then(response => {
        return new File([response.data], 'inputImage')
    }).then(file => {
        form.append('image', file);
        form.append('method', 'contrast');
        axios.post('/api/enhancement', form).then(function (response) {
            console.log('Upload Success');
            revert_images.push(document.getElementById('result_image').src);
            document.getElementById('revert').disabled = false;
            document.getElementById('result_image').src = response.data.result;
        }).catch(function (error) {
            alert('Upload Failed');
        }).finally(() => {
            this.disabled = false;
            document.getElementById('contrast_spinner').classList.add('d-none');
        });
    })
})
document.getElementById('grayscale_button').addEventListener('click', function () {
    this.disabled = true;
    document.getElementById('grayscale_spinner').classList.remove('d-none');
    // upload blob image url to server
    let form = new FormData(document.getElementById('grayscale_options'));
    let blob_url = document.getElementById('result_image').src;
    // get the image from object url and send to server
    axios.get(blob_url, { responseType: 'blob' }).then(response => {
        return new File([response.data], 'inputImage')
    }).then(file => {
        form.append('image', file);
        form.append('method', 'grayscale');
        axios.post('/api/enhancement', form).then(function (response) {
            console.log('Upload Success');
            revert_images.push(document.getElementById('result_image').src);
            document.getElementById('revert').disabled = false;
            document.getElementById('result_image').src = response.data.result;
        }).catch(function (error) {
            alert('Upload Failed');
        }).finally(() => {
            this.disabled = false;
            document.getElementById('grayscale_spinner').classList.add('d-none');
        });
    })
})
document.getElementById('gamma_button').addEventListener('click', function () {
    this.disabled = true;
    document.getElementById('gamma_spinner').classList.remove('d-none');
    // upload blob image url to server
    let form = new FormData(document.getElementById('gamma_options'));
    let blob_url = document.getElementById('result_image').src;
    // get the image from object url and send to server
    axios.get(blob_url, { responseType: 'blob' }).then(response => {
        return new File([response.data], 'inputImage')
    }).then(file => {
        form.append('image', file);
        form.append('method', 'gamma');
        axios.post('/api/enhancement', form).then(function (response) {
            console.log('Upload Success');
            revert_images.push(document.getElementById('result_image').src);
            document.getElementById('revert').disabled = false;
            document.getElementById('result_image').src = response.data.result;
        }).catch(function (error) {
            alert('Upload Failed');
        }).finally(() => {
            this.disabled = false;
            document.getElementById('gamma_spinner').classList.add('d-none');
        });
    })
})
document.getElementById('colorbalance_button').addEventListener('click', function () {
    this.disabled = true;
    document.getElementById('colorbalance_spinner').classList.remove('d-none');
    // upload blob image url to server
    let form = new FormData(document.getElementById('colorbalance_options'));
    let blob_url = document.getElementById('result_image').src;
    // get the image from object url and send to server
    axios.get(blob_url, { responseType: 'blob' }).then(response => {
        return new File([response.data], 'inputImage')
    }).then(file => {
        form.append('image', file);
        form.append('method', 'colorbalance');
        axios.post('/api/enhancement', form).then(function (response) {
            console.log('Upload Success');
            revert_images.push(document.getElementById('result_image').src);
            document.getElementById('revert').disabled = false;
            document.getElementById('result_image').src = response.data.result;
        }).catch(function (error) {
            alert('Upload Failed');
        }).finally(() => {
            this.disabled = false;
            document.getElementById('colorbalance_spinner').classList.add('d-none');
        });
    })
})
document.getElementById('shadowremove_button').addEventListener('click', async function () {
    this.disabled = true;
    document.getElementById('shadowremove_spinner').classList.remove('d-none');
    var n_iters = document.getElementById('shadowremove_n_iter').value
    try {
        for (let index = 0; index < n_iters; index++) {
            // upload blob image url to server
            let form = new FormData(document.getElementById('shadowremove_options'));
            let blob_url = document.getElementById('result_image').src;
            // get the image from object url and send to server
            let response = await axios.get(blob_url, { responseType: 'blob' })
            let file = new File([response.data], 'inputImage')
            form.append('image', file);
            form.append('method', 'shadowremove');
            response = await axios.post('/api/enhancement', form)
            console.log('Upload Success');
            revert_images.push(document.getElementById('result_image').src);
            document.getElementById('revert').disabled = false;
            document.getElementById('result_image').src = response.data.result;
        }
    } catch (error) {
        alert('Upload Failed');
    } finally {
        this.disabled = false;
        document.getElementById('shadowremove_spinner').classList.add('d-none');
    }
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

document.getElementById('revert').addEventListener('click', function () {
    if (revert_images.length > 0) {
        document.getElementById('result_image').src = revert_images.pop();
    }
    if (revert_images.length == 0)
        this.disabled = true;
});

document.getElementById('next_step').addEventListener('click', function () {
    image_url = document.getElementById('result_image').src;
    window.location.href = '/ocr?image_url=' + image_url;
});