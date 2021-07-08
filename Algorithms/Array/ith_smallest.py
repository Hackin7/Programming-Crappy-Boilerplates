import random
def generate():
    return [random.randint(0,1000) for i in range(10)]
    
def select(arr, s, e, i):
    if not s<e: 
        return arr[s]
    ### Partition
    pivot,l,r = arr[s],s+1,e
    while l <= r:
        while l <= r and arr[l] <= pivot:
            l+=1
        while l <= r and arr[r] >= pivot:
            r-=1
        if l<=r:
            temp = arr[l]
            arr[l] = arr[r]
            arr[r] = temp
    ### Swapping in Pivot
    temp = arr[s]
    arr[s] = arr[r]
    arr[r] = temp
    ### Recursive Case
    print(arr,s,l,r,e)
    pivot_rank = r - s + 1
    if pivot_rank == i:
        return pivot
    elif i < pivot_rank:
        return select(arr,s,r-1,i)
    elif pivot_rank < i:
        return select(arr,r+1,e,i - pivot_rank)

arr = generate()
print(arr)
print(select(arr,0,len(arr)-1, 1))
#print(arr)

