FROM androidsdk/android-27
#thyrlian/android-sdk

RUN apt install -y ant

WORKDIR /usr/src/app

RUN apt install -y wget
# https://androidsdkmanager.azurewebsites.net/SDKPlatform
run wget https://dl.google.com/android/repository/platform-29_r01.zip && unzip platform-29_r01.zip && rm platform-29_r01.zip && mv android-10 platform
# https://androidsdkmanager.azurewebsites.net/Buildtools run wget https://dl.google.com/android/repository/build-tools_r30.0.1-linux.zip && unzip build-tools_r30.0.1-linux.zip && rm build-tools_r30.0.1-linux.zip && mv android-11 build-tools

# Copy all items
COPY . .

#Expose ports if needed
# EXPOSE 80 5000
#  docker run -dp 3000:3000 getting-started

## Entry points
# To run the container, docker build <directory>, docker run -it <image_id>
ENTRYPOINT  ["bash", "-c", "bash"]
CMD ["bash -c bash"]
