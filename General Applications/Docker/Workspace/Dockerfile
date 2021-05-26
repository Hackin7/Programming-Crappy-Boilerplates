FROM kalilinux/kali-rolling

# Copy all items
WORKDIR /usr/src/app
COPY . .

# Get list of items
RUN ls

#Expose ports if needed
# EXPOSE 80 5000
#  docker run -dp 3000:3000 getting-started

## Entry points
# To run the container, docker build <directory>, docker run -it <image_id>
ENTRYPOINT  ["bash", "-c", "bash"]
CMD ["bash -c bash"]
