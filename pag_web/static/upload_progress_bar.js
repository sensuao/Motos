
  var form = document.getElementById('upload-form');
  var progress = document.getElementById('upload-progress');

  form.addEventListener('submit', function(event) {
      event.preventDefault();  // Evita la acción predeterminada del formulario

      var fileInput = document.querySelector('input[type="file"]');
      var file = fileInput.files[0];
      var formData = new FormData();

      formData.append('file', file);

      var xhr = new XMLHttpRequest();
      xhr.open('POST', '/upload', true);

      // Actualiza la barra de progreso durante la carga
      xhr.upload.onprogress = function(e) {
          if (e.lengthComputable) {
              var percent = (e.loaded / e.total) * 100;
              progress.value = percent;
          }
      };

      // Maneja la respuesta del servidor después de la carga
      xhr.onload = function() {
          if (xhr.status === 200) {
              // Carga exitosa
              progress.value = 100;
              window.location.href = '{{ url_for("index") }}';
          } else {
              // Error de carga
              alert('Error during file upload. Please try again.');
          }
      };

      // Envía la solicitud al servidor
      xhr.send(formData);
  });
