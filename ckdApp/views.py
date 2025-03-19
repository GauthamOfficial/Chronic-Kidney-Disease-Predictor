import pickle
import numpy as np
from django.shortcuts import render, redirect
from django.views import View
from django.urls import reverse_lazy
from sklearn.preprocessing import StandardScaler
from .forms import ckdForm

class dataUploadView(View):
    form_class = ckdForm
    success_url = reverse_lazy('success')
    failure_url = reverse_lazy('fail')
    template_name = 'create.html'

    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST, request.FILES)

        if form.is_valid():
            form.save()

            # Retrieve input values and convert them to float
            try:
                data_bgr = float(request.POST.get('Blood_Glucose_Random', 0))
                data_bu = float(request.POST.get('Blood_Urea', 0))
                data_sc = float(request.POST.get('Serum_Creatine', 0))
                data_pcv = float(request.POST.get('Packed_cell_volume', 0))
                data_wc = float(request.POST.get('White_blood_count', 0))
            except ValueError:
                return redirect(self.failure_url)  # Redirect if input is invalid

            # Load trained model
            filename = 'finalized_model_DTC3.sav'
            classifier = pickle.load(open(filename, 'rb'))

            # Load pre-fitted StandardScaler
            scaler_filename = 'sc.pkl'  # Ensure you have saved the scaler earlier
            sc = pickle.load(open(scaler_filename, 'rb'))

            # Prepare input array (2D array required for StandardScaler)
            data = np.array([[data_bgr, data_bu, data_sc, data_pcv, data_wc]])

            # Apply Standard Scaling
            data_scaled = sc.transform(data)

            # Predict the output
            out = classifier.predict(data_scaled)

            # Render the success page with results
            return render(request, "succ_msg.html", {
                'data_bgr': data_bgr,
                'data_bu': data_bu,
                'data_sc': data_sc,
                'data_pcv': data_pcv,
                'data_wc': data_wc,
                'out': out
            })

        else:
            return redirect(self.failure_url)

