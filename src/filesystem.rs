// #![cfg_attr(not(test), no_std)]

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum FileSystemResult<T: Copy + Clone> {
    Ok(T),
    Err(FileSystemError),
}

impl<T: Copy + Clone> FileSystemResult<T> {
    pub fn unwrap(&self) -> T {
        match self {
            FileSystemResult::Ok(v) => *v,
            FileSystemResult::Err(e) => panic!("Error: {e:?}"),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum FileSystemError {
    FileNotFound,
    FileNotOpen,
    NotOpenForRead,
    NotOpenForWrite,
    TooManyOpen,
    TooManyFiles,
    AlreadyOpen,
    DiskFull,
    FileTooBig,
    FilenameTooLong,
}

#[derive(Debug, Copy, Clone)]
pub struct FileInfo<const MAX_BLOCKS: usize, const BLOCK_SIZE: usize> {
    inode: Inode<MAX_BLOCKS, BLOCK_SIZE>,
    inode_num: usize,
    current_block: usize,
    offset: usize,
    writing: bool,
    block_buffer: [u8; BLOCK_SIZE],
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Inode<const MAX_BLOCKS: usize, const BLOCK_SIZE: usize> {
    bytes_stored: u16,
    blocks: [u8; MAX_BLOCKS],
}

const INODE_FULL_BLOCK: usize = 0;
const DATA_FULL_BLOCK: usize = INODE_FULL_BLOCK + 1;
const INODE_TABLE_START: usize = DATA_FULL_BLOCK + 1;

#[derive(core::fmt::Debug)]
pub struct FileSystem<
    const MAX_OPEN: usize,
    const BLOCK_SIZE: usize,
    const NUM_BLOCKS: usize,
    const MAX_FILE_BLOCKS: usize,
    const MAX_FILE_BYTES: usize,
    const MAX_FILES_STORED: usize,
    const MAX_FILENAME_BYTES: usize,
> {
    open: [Option<FileInfo<MAX_FILE_BLOCKS, BLOCK_SIZE>>; MAX_OPEN],
    disk: ramdisk::RamDisk<BLOCK_SIZE, NUM_BLOCKS>,
    block_buffer: [u8; BLOCK_SIZE],
    file_content_buffer: [u8; MAX_FILE_BYTES],
    open_inodes: [bool; MAX_FILES_STORED],
}

impl<
    const MAX_OPEN: usize,
    const BLOCK_SIZE: usize,
    const NUM_BLOCKS: usize,
    const MAX_FILE_BLOCKS: usize,
    const MAX_FILE_BYTES: usize,
    const MAX_FILES_STORED: usize,
    const MAX_FILENAME_BYTES: usize,
>
FileSystem<
    MAX_OPEN,
    BLOCK_SIZE,
    NUM_BLOCKS,
    MAX_FILE_BLOCKS,
    MAX_FILE_BYTES,
    MAX_FILES_STORED,
    MAX_FILENAME_BYTES,
>
{
    pub fn new(disk: ramdisk::RamDisk<BLOCK_SIZE, NUM_BLOCKS>) -> Self {
        assert_eq!(MAX_FILE_BYTES, MAX_FILE_BLOCKS * BLOCK_SIZE);
        assert!(NUM_BLOCKS <= u8::MAX as usize);
        assert!(MAX_FILE_BYTES <= u16::MAX as usize);
        let block_bits = BLOCK_SIZE * 8;
        assert!(MAX_FILES_STORED <= block_bits);
        assert!(MAX_FILES_STORED <= u16::MAX as usize);
        let result = Self {
            open: [None; MAX_OPEN],
            disk,
            block_buffer: [0; BLOCK_SIZE],
            file_content_buffer: [0; MAX_FILE_BYTES],
            open_inodes: [false; MAX_FILES_STORED],
        };
        assert!(result.num_inode_blocks() * 2 < NUM_BLOCKS);
        assert!(result.num_data_blocks() <= block_bits);
        assert_eq!(
            result.num_data_blocks() + result.num_inode_blocks() + 2,
            NUM_BLOCKS
        );
        assert!(result.num_inode_entries() <= u16::MAX as usize);
        assert!(result.num_inode_blocks() <= MAX_FILE_BLOCKS);
        result
    }

    pub fn assert_block(&self, block: usize, offset: usize, block_segment: &[u8]) {
        assert!(block < self.disk.num_blocks());
        let mut bytes = [0; BLOCK_SIZE];
        self.disk.read(block, &mut bytes);
        assert_eq!(block_segment, &bytes[offset..offset + block_segment.len()]);
    }

    pub fn max_file_size(&self) -> usize {
        // 8 * 64 = 512
        MAX_FILE_BLOCKS * BLOCK_SIZE
    }

    pub fn num_inode_bytes(&self) -> usize {
        // 10
        2 + MAX_FILE_BLOCKS
    }

    pub fn inodes_per_block(&self) -> usize {
        // 64 / 10 = 6
        BLOCK_SIZE / self.num_inode_bytes()
    }

    pub fn num_inode_blocks(&self) -> usize {
        // 32 / 6 = 5
        MAX_FILES_STORED / self.inodes_per_block()
    }

    pub fn num_data_blocks(&self) -> usize {
        // 255 - 5 - 2 = 248
        NUM_BLOCKS - self.num_inode_blocks() - 2
    }

    pub fn num_inode_entries(&self) -> usize {
        // 6 * 5 * 10 = 300
        self.inodes_per_block() * self.num_inode_blocks() * self.num_inode_bytes()
    }

    pub fn first_data_block(&self) -> usize {
        // 2 + 5 = 7
        2 + self.num_inode_blocks()
    }

    // filename per block
    pub fn max_filenames_per_block(&self) -> usize {
        // 64 / 8 = 8
        BLOCK_SIZE / MAX_FILENAME_BYTES
    }

    // convert inode to bytes
    // byte 0: low-order bits of bytes_stored
    // byte 1: high-order bits of bytes_stored
    // bytes 2..MAX_BLOCKS + 2: blocks
    pub fn inode_to_bytes(&self, inode: Inode<MAX_FILE_BLOCKS, BLOCK_SIZE>, buffer: &mut [u8; BLOCK_SIZE]) {
        buffer[0..2].copy_from_slice(&inode.bytes_stored.to_le_bytes());
        buffer[2..2 + MAX_FILE_BLOCKS].copy_from_slice(&inode.blocks);
    }

    // convert bytes to inode
    pub fn bytes_to_inode(&self, buffer: [u8; BLOCK_SIZE], offset: usize) -> Inode<MAX_FILE_BLOCKS, BLOCK_SIZE> {
        let bytes_stored = u16::from_le_bytes([buffer[offset], buffer[offset + 1]]);
        let mut blocks: [u8; MAX_FILE_BLOCKS] = [0; MAX_FILE_BLOCKS];
        blocks.copy_from_slice(&buffer[offset + 2..offset + 2 + MAX_FILE_BLOCKS]);
        Inode { bytes_stored, blocks }
    }

    // write inode + inode_num
    // write to inode_block and set bit at block 0 and 1
    pub fn write_inode(&mut self, inode_num: usize, inode: Inode<MAX_FILE_BLOCKS, BLOCK_SIZE>) {
        let mut inode_bytes = [0u8; BLOCK_SIZE];
        self.inode_to_bytes(inode, &mut inode_bytes);

        let offset = inode_num * self.num_inode_bytes();
        for i in 0..self.num_inode_bytes() {
            let byte_idx = offset + i;
            let block_num = INODE_TABLE_START + byte_idx / BLOCK_SIZE;
            let byte_in_block_idx = byte_idx % BLOCK_SIZE;
            self.write_byte_to_block(block_num, byte_in_block_idx, inode_bytes[i]);
        }



        // set bit to block 0 (inode table)
        self.write_bit_to_block(0, inode_num, true);

        // set bit to block 1(data blocks table)
        for i in 0..MAX_FILE_BLOCKS {
            if inode.blocks[i] != 0 {
                self.write_bit_to_block(1, inode.blocks[i] as usize, true);
            }
        }
    }

    // read inode from inode_num
    pub fn read_inode(&self, inode_num: usize) -> Inode<MAX_FILE_BLOCKS, BLOCK_SIZE> {
        let mut buffer = [0; BLOCK_SIZE];
        let offset = inode_num * self.num_inode_bytes();
        // self.bytes_to_inode(buffer, offset)
        for i in 0..self.num_inode_bytes() {
            let byte_idx = offset + i;
            let block_num = INODE_TABLE_START + byte_idx / BLOCK_SIZE;
            let byte_in_block_idx = byte_idx % BLOCK_SIZE;
            buffer[i] = self.read_byte_from_block(block_num, byte_in_block_idx);
        }
        self.bytes_to_inode(buffer, 0)
    }

    // write a bit to a block
    pub fn write_bit_to_block(&mut self, block_num: usize, bit: usize, value: bool) {
        let mut buffer: [u8; BLOCK_SIZE] = [0; BLOCK_SIZE];
        self.disk.read(block_num, &mut buffer);
        let byte = bit / 8;
        let bit = bit % 8;
        if value {
            buffer[byte] |= 1 << bit;
        } else {
            buffer[byte] &= !(1 << bit);
        }
        self.disk.write(block_num, &buffer);
    }

    // read a bit value from a block
    pub fn read_bit_from_block(&self, block_num: usize, bit: usize) -> bool {
        let mut buffer: [u8; BLOCK_SIZE] = [0; BLOCK_SIZE];
        self.disk.read(block_num, &mut buffer);
        let byte = bit / 8;
        let bit = bit % 8;
        (buffer[byte] & (1 << bit)) != 0
    }

    pub fn read_byte_from_block(&self, block_num: usize, byte: usize) -> u8 {
        let mut buffer: [u8; BLOCK_SIZE] = [0; BLOCK_SIZE];
        self.disk.read(block_num, &mut buffer);
        buffer[byte]
    }

    pub fn write_byte_to_block(&mut self, block_num: usize, byte: usize, value: u8) {
        let mut buffer: [u8; BLOCK_SIZE] = [0; BLOCK_SIZE];
        self.disk.read(block_num, &mut buffer);
        buffer[byte] = value;
        self.disk.write(block_num, &buffer);
    }

    // find first free block -> return block_num
    pub fn find_free_block(&self, start: usize) -> FileSystemResult<usize> {
        for i in start..NUM_BLOCKS {
            if !self.read_bit_from_block(1, i) {
                return FileSystemResult::Ok(i);
            }
        }
        FileSystemResult::Err(FileSystemError::DiskFull)
    }

    // find free inode
    pub fn find_free_inode(&self) -> FileSystemResult<usize> {
        for i in 0..MAX_FILES_STORED {
            if !self.read_bit_from_block(0, i) {
                return FileSystemResult::Ok(i);
            }
        }
        FileSystemResult::Err(FileSystemError::TooManyFiles)
    }

    pub fn create_directory_if_not_exists(&mut self) {
        // check if inode 0 is in use
        if !self.read_bit_from_block(0, 0) {
            // use block 0, 1 + num_inodes_blocks
            for i in 0..self.first_data_block() {
                self.write_bit_to_block(1, i, true);
            }
            // create an inode for the directory
            let mut inode: Inode<MAX_FILE_BLOCKS, BLOCK_SIZE> = Inode {
                bytes_stored: 8,
                blocks: [0; MAX_FILE_BLOCKS],
            };
            let inode_num = 0;
            inode.blocks[0] = self.first_data_block() as u8;
            self.write_inode(inode_num, inode);
        }
    }

    // write buffer to block from offset
    pub fn write_block_offset(&mut self, block_num: usize, buffer: &[u8], offset: usize) {

        let mut block_buffer = [0; BLOCK_SIZE];
        self.disk.read(block_num, &mut block_buffer);
        block_buffer[offset..offset + buffer.len()].copy_from_slice(buffer);
        self.disk.write(block_num, &block_buffer);
    }

    // read a block to buffer with offset
    pub fn read_block_offset(&self, block_num: usize, buffer: &mut [u8], offset: usize) {
        let mut block_buffer = [0; BLOCK_SIZE];
        self.disk.read(block_num, &mut block_buffer);
        buffer.copy_from_slice(&block_buffer[offset..offset + buffer.len()]);
    }

    // write file name to directory
    // add bytes to inode, if last block is full, add a new block
    pub fn write_filename_to_directory(&mut self, filename: &str, inode_num: usize) -> FileSystemResult<()> {
        let mut inode = self.read_inode(0);
        let mut filename_bytes_buffer: [u8; MAX_FILENAME_BYTES] = [0; MAX_FILENAME_BYTES];
        let filename_bytes = filename.as_bytes();
        filename_bytes_buffer[0..filename_bytes.len()].copy_from_slice(filename_bytes);

        let block_idx = inode_num / self.max_filenames_per_block();
        if block_idx <= self.find_last_block_idx(inode) {
            // write to current block
            let buffer_offset = (inode_num % self.max_filenames_per_block()) * MAX_FILENAME_BYTES;
            self.write_block_offset(inode.blocks[0] as usize, &filename_bytes_buffer, buffer_offset);
            inode.bytes_stored += MAX_FILENAME_BYTES as u16;
            self.write_inode(0, inode);

            // if current if full, create new block
            if inode.bytes_stored % BLOCK_SIZE as u16 == 0 {
                let new_block = match self.find_free_block(self.first_data_block()) {
                    FileSystemResult::Ok(new_block) => new_block,
                    FileSystemResult::Err(e) => return FileSystemResult::Err(e),
                };
                inode.blocks[self.find_last_block_idx(inode) + 1] = new_block as u8;
                self.write_inode(0, inode);
            }

            return FileSystemResult::Ok(());
        }

        // write to new block
        let new_block = match self.find_free_block(self.first_data_block()) {
            FileSystemResult::Ok(new_block) => new_block,
            FileSystemResult::Err(e) => return FileSystemResult::Err(e),
        };

        let buffer_offset = (inode_num % self.max_filenames_per_block()) * MAX_FILENAME_BYTES;
        self.write_block_offset(new_block, &filename_bytes_buffer, buffer_offset);
        inode.bytes_stored += MAX_FILENAME_BYTES as u16;
        inode.blocks[self.find_last_block_idx(inode) + 1] = new_block as u8;
        self.write_inode(0, inode);
        FileSystemResult::Ok(())
    }

    pub fn find_inode(&self, filename: &str) -> FileSystemResult<usize> {
        // To find an inode:
        // Load the directory file.
        // Search the directory file for the filename.
        // If the filename is not found, return an error.
        // Return the inode number associated with the filename.

        // start the code
        if filename.len() > MAX_FILENAME_BYTES {
            return FileSystemResult::Err(FileSystemError::FilenameTooLong);
        }
        let inode = self.read_inode(0);
        let mut filename_bytes_buffer: [u8; MAX_FILENAME_BYTES] = [0; MAX_FILENAME_BYTES];
        let filename_bytes = filename.as_bytes();
        filename_bytes_buffer[0..filename_bytes.len()].copy_from_slice( filename_bytes);
        let mut inode_num = 0;
        let mut block_idx = 0;
        loop {
            let buffer_offset = (inode_num % self.max_filenames_per_block()) * MAX_FILENAME_BYTES;
            let mut directory_buffer = [0u8; MAX_FILENAME_BYTES];
            self.read_block_offset(inode.blocks[block_idx] as usize, &mut directory_buffer, buffer_offset);

            if directory_buffer == filename_bytes_buffer {
                return FileSystemResult::Ok(inode_num);
            }
            inode_num += 1;
            block_idx = inode_num / self.max_filenames_per_block();
            if block_idx > self.find_last_block_idx(inode) {
                break;
            }
        }
        FileSystemResult::Err(FileSystemError::FileNotFound)
    }

    pub fn find_open_fd(&self) -> FileSystemResult<usize> {
        // To find an open file descriptor:
        // Search the open file table for an open file descriptor.
        // If no open file descriptors are found, return an error.
        // Return the file descriptor.

        // start the code
        for i in 0..MAX_OPEN {
            if self.open[i].is_none() {
                return FileSystemResult::Ok(i);
            }
        }
        FileSystemResult::Err(FileSystemError::TooManyOpen)
    }

    // check is_open, return MAX_OPEN if not found
    pub fn is_open(&self, inode_num: usize) -> usize {
        for i in 0..MAX_OPEN {
            if let Some(file_info) = self.open[i] {
                if file_info.inode_num == inode_num {
                    return i;
                }
            }
        }
        MAX_OPEN
    }

    // find last block of inode
    pub fn find_last_block_idx(&self, inode: Inode<MAX_FILE_BLOCKS, BLOCK_SIZE>) -> usize {
        for i in 1..MAX_FILE_BLOCKS {
            if inode.blocks[i] == 0 {
                return i - 1;
            }
        }
        MAX_FILE_BLOCKS - 1
    }

    pub fn find_last_block(&self, inode: Inode<MAX_FILE_BLOCKS, BLOCK_SIZE>) -> usize {
        return inode.blocks[self.find_last_block_idx(inode)] as usize;
    }

    pub fn open_read(&mut self, filename: &str) -> FileSystemResult<usize> {
        // To open a file to read:
        // Load the directory file, and find the file’s inode.
        // If the file is not present in the directory, return an error.
        // If the inode is already open, return an error.
        // Create a file table entry (a FileInfo object) for the newly opened file.
        // Read in the first block of the newly opened file into the file table entry’s buffer.
        // Mark the inode as open in open_inodes
        // Return the file descriptor, that is, the index of the file table used for its FileInfo.   

        // start the code
        let inode_num = match self.find_inode(filename) {
            FileSystemResult::Ok(inode_num) => inode_num,
            FileSystemResult::Err(e) => return FileSystemResult::Err(e),
        };

        if self.open_inodes[inode_num] {
            return FileSystemResult::Err(FileSystemError::AlreadyOpen);
        }
        let fd = match self.find_open_fd() {
            FileSystemResult::Ok(fd) => fd,
            FileSystemResult::Err(e) => return FileSystemResult::Err(e),
        };
        let inode = self.read_inode(inode_num);

        let mut file_info = FileInfo {
            inode,
            inode_num,
            current_block: 0,
            offset: 0,
            writing: false,
            block_buffer: [0; BLOCK_SIZE],
        };
        self.read_block_offset(inode.blocks[0] as usize, &mut file_info.block_buffer, 0);
        self.open[fd] = Some(file_info);
        self.open_inodes[inode_num] = true;
        FileSystemResult::Ok(fd)
    }

    pub fn open_create(&mut self, filename: &str) -> FileSystemResult<usize> {
        // To create a file:

        // Create the directory file, if it does not already exist.
        // To check if it exists, see if inode 0 is in use.
        // If it does not exist:
        // Set the bit for inode 0 to 1.
        // Select its first data block.
        // Create an inode for the directory, and save it in the inode table.
        // If the file already has an inode:
        // Use the current inode.
        // Reset its stored-bytes and current-block to a state as if it were newly created.
        // Clear the in-use bits for its existing data blocks, except for the first data block. We will continue to use that block as we start the write.
        // Otherwise:
        // Select an inode number.
        // Select a data block number for the first block.
        // Create an inode for the new file, and save it in the inode table.
        // Update the directory file with an entry for the new file.
        // Create a file table entry for the newly created file, and return the file descriptor.

        // start the code

        if filename.len() > MAX_FILENAME_BYTES {
            return FileSystemResult::Err(FileSystemError::FilenameTooLong);
        }

        self.create_directory_if_not_exists();


        // get inode_num from file_name
        // create if needed
        // if exist, reset bytes_stored and blocks from 1
        let inode_num: usize = match self.find_inode(filename) {
            FileSystemResult::Ok(inode_num) => {
                let mut inode = self.read_inode(inode_num);
                inode.bytes_stored = 0;
                // reset ffrom block 1
                for i in 1..MAX_FILE_BLOCKS {
                    if inode.blocks[i] != 0 {
                        let new_buffer = [0u8; BLOCK_SIZE];
                        self.disk.write(inode.blocks[i] as usize, &new_buffer);
                        self.write_bit_to_block(1, inode.blocks[i] as usize, false);
                        inode.blocks[i] = 0;
                    }
                }
                // reset inode.blocks[0]
                let new_buffer = [0u8; BLOCK_SIZE];
                self.disk.write(inode.blocks[0] as usize, &new_buffer);

                self.write_inode(inode_num, inode);
                inode_num
            },
            FileSystemResult::Err(_) => {
                let inode_num = match self.find_free_inode() {
                    FileSystemResult::Ok(inode_num) => inode_num,
                    FileSystemResult::Err(e) => return FileSystemResult::Err(e),
                };
                self.write_filename_to_directory(filename, inode_num);
                // create inode
                let mut inode = Inode {
                    bytes_stored: 0,
                    blocks: [0; MAX_FILE_BLOCKS],
                };
                // select a data block number for the first block
                let data_block = match self.find_free_block(self.first_data_block()) {
                    FileSystemResult::Ok(data_block) => data_block,
                    FileSystemResult::Err(e) => return FileSystemResult::Err(e),
                };
                inode.blocks[0] = data_block as u8;
                self.write_inode(inode_num, inode);

                inode_num
            }
        };

        if inode_num == MAX_FILES_STORED {
            return FileSystemResult::Err(FileSystemError::TooManyFiles);
        }

        let mut open_idx = self.is_open(inode_num);
        // not open yet, open it
        if open_idx == MAX_OPEN {
            // find a free open slot
            open_idx = match self.find_open_fd() {
                FileSystemResult::Ok(open_idx) => open_idx,
                FileSystemResult::Err(e) => return FileSystemResult::Err(e),
            };
            let inode = self.read_inode(inode_num);
            let file_info = FileInfo {
                inode,
                inode_num,
                current_block: 0,
                offset: 0,
                writing: true,
                block_buffer: [0; BLOCK_SIZE],
            };
            self.open[open_idx] = Some(file_info);
            self.open_inodes[inode_num] = true;
        }

        FileSystemResult::Ok(open_idx)
    }

    pub fn open_append(&mut self, filename: &str) -> FileSystemResult<usize> {
        let inode_num = match self.find_inode(filename) {
            FileSystemResult::Ok(inode_num) => inode_num,
            FileSystemResult::Err(e) => return FileSystemResult::Err(e),
        };
        if self.open_inodes[inode_num] {
            return FileSystemResult::Err(FileSystemError::AlreadyOpen);
        }
        let fd = match self.find_open_fd() {
            FileSystemResult::Ok(fd) => fd,
            FileSystemResult::Err(e) => return FileSystemResult::Err(e),
        };

        let inode = self.read_inode(inode_num);
        let cur_block_idx = self.find_last_block_idx(inode);
        let cur_block = inode.blocks[cur_block_idx] as usize;
        let mut file_info = FileInfo {
            inode,
            inode_num,
            current_block: cur_block_idx,
            offset: (inode.bytes_stored % BLOCK_SIZE as u16) as usize,
            writing: true,
            block_buffer: [0; BLOCK_SIZE],
        };
        self.read_block_offset(cur_block, &mut file_info.block_buffer, 0);
        self.open[fd] = Some(file_info);
        self.open_inodes[inode_num] = true;
        FileSystemResult::Ok(fd)
    }

    pub fn close(&mut self, fd: usize) -> FileSystemResult<()> {
        // close the fd
        let file_info = match self.open[fd] {
            Some(file_info) => file_info,
            None => return FileSystemResult::Err(FileSystemError::FileNotOpen),
        };
        if file_info.writing {
            let mut inode = file_info.inode;
            inode.bytes_stored = file_info.offset as u16;
            self.disk.write(inode.blocks[file_info.current_block] as usize, &file_info.block_buffer);
        }
        self.open[fd] = None;
        self.open_inodes[file_info.inode_num] = false;
        FileSystemResult::Ok(())
    }

    // return number of bytes read
    pub fn read(&mut self, fd: usize, buffer: &mut [u8]) -> FileSystemResult<usize> {
        // print file info
        let mut file_info = match self.open[fd] {
            Some(file_info) => file_info,
            None => return FileSystemResult::Err(FileSystemError::FileNotOpen),
        };
        if file_info.writing {
            return FileSystemResult::Err(FileSystemError::NotOpenForRead);
        }


        let mut buffer_offset = 0;
        let bytes_stored = file_info.inode.bytes_stored as usize;
        while buffer_offset < buffer.len() {
            let file_open_offset = file_info.offset;
            let remaining_block_space: usize;
            let rem_buffer = buffer.len() - buffer_offset;
            if (bytes_stored / BLOCK_SIZE) == file_info.current_block {
                remaining_block_space = bytes_stored % BLOCK_SIZE - file_open_offset;
                if remaining_block_space == 0 {
                    break;
                }
            } else {
                remaining_block_space = BLOCK_SIZE - file_open_offset;
            }

            if rem_buffer <= remaining_block_space {
                buffer[buffer_offset..buffer_offset + rem_buffer].copy_from_slice(&file_info.block_buffer[file_open_offset..file_open_offset + rem_buffer]);
                buffer_offset += rem_buffer;
                file_info.offset += rem_buffer;
            } else {
                buffer[buffer_offset..buffer_offset + remaining_block_space].copy_from_slice(&file_info.block_buffer[file_open_offset..file_open_offset + remaining_block_space]);
                buffer_offset += remaining_block_space;
                file_info.offset += remaining_block_space;
            }

            if file_info.offset == BLOCK_SIZE {
                file_info.offset = 0;
                file_info.current_block += 1;
                self.read_block_offset(file_info.inode.blocks[file_info.current_block] as usize, &mut file_info.block_buffer, 0);
            }
        }

        self.open[fd] = Some(file_info);

        FileSystemResult::Ok(buffer_offset)
    }

    pub fn write(&mut self, fd: usize, buffer: &[u8]) -> FileSystemResult<()> {

        let mut file_info = match self.open[fd] {
            Some(file_info) => file_info,
            None => return FileSystemResult::Err(FileSystemError::FileNotOpen),
        };
        if !file_info.writing {
            return FileSystemResult::Err(FileSystemError::NotOpenForWrite);
        }

        // let inode = file_info.inode;

        let mut buffer_offset = 0;

        while buffer_offset < buffer.len()  {
            let file_open_offset = file_info.offset;
            let mut current_block_buffer = file_info.block_buffer;
            let remaining_block_space = BLOCK_SIZE - file_open_offset;
            let remaining_buffer = buffer.len() - buffer_offset;

            if remaining_buffer <= remaining_block_space {
                // write to current block with some space left
                current_block_buffer[file_open_offset..file_open_offset + remaining_buffer].copy_from_slice(&buffer[buffer_offset..buffer_offset + remaining_buffer]);
                self.write_block_offset(file_info.inode.blocks[file_info.current_block] as usize, &current_block_buffer, 0);
                file_info.offset += remaining_buffer;
                buffer_offset += remaining_buffer;
                file_info.block_buffer = current_block_buffer;
                file_info.inode.bytes_stored += remaining_buffer as u16;
                if file_info.offset == BLOCK_SIZE {
                    file_info.offset = 0;
                    file_info.current_block += 1;
                    if file_info.current_block == MAX_FILE_BLOCKS {
                        return FileSystemResult::Err(FileSystemError::FileTooBig);
                    }
                    let next_block = match self.find_free_block(self.first_data_block()) {
                        FileSystemResult::Ok(next_block) => next_block,
                        FileSystemResult::Err(e) => return FileSystemResult::Err(e),
                    };
                    file_info.inode.blocks[file_info.current_block] = next_block as u8;
                    self.write_inode(file_info.inode_num, file_info.inode);
                    self.read_block_offset(next_block, &mut file_info.block_buffer, 0);
                }
                self.write_inode(file_info.inode_num, file_info.inode);
            } else {
                // write to current block until end
                current_block_buffer[file_open_offset..file_open_offset + remaining_block_space].copy_from_slice(&buffer[buffer_offset..buffer_offset + remaining_block_space]);
                self.write_block_offset(file_info.inode.blocks[file_info.current_block] as usize, &current_block_buffer, 0);
                buffer_offset += remaining_block_space;
                // find next block
                let next_block = match self.find_free_block(self.first_data_block()) {
                    FileSystemResult::Ok(next_block) => next_block,
                    FileSystemResult::Err(e) => return FileSystemResult::Err(e),
                };
                file_info.offset = 0;
                file_info.current_block += 1;
                if file_info.current_block == MAX_FILE_BLOCKS {
                    return FileSystemResult::Err(FileSystemError::FileTooBig);
                }
                file_info.inode.blocks[file_info.current_block] = next_block as u8;
                file_info.inode.bytes_stored += remaining_block_space as u16;
                self.write_inode(file_info.inode_num, file_info.inode);

                self.read_block_offset(next_block, &mut file_info.block_buffer, 0);
            }
        }

        // print first 16 bytes of file_info.block_buffer


        // inode.bytes_stored += buffer.len() as u16;
        // self.write_inode(file_info.inode_num, inode);


        self.open[fd] = Some(file_info);
        FileSystemResult::Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK_SIZE: usize = 64;
    const MAX_FILES_STORED: usize = 32;

    fn make_small_fs() -> FileSystem<16, 64, 255, 8, 512, 32, 8> {
        FileSystem::new(ramdisk::RamDisk::new())
    }

    #[test]
    fn test_short_write() {
        let mut sys = make_small_fs();
        let f1 = sys.open_create("one.txt").unwrap();
        sys.assert_block(0, 0, &[3, 0]);
        sys.assert_block(1, 0, &[255, 1, 0]);
        sys.assert_block(2, 0, &[16, 0, 7]);
        sys.assert_block(2, 10, &[0, 0, 8]);
        sys.assert_block(7, 0, &[0, 0, 0, 0, 0, 0, 0, 0, 111, 110, 101, 46, 116, 120, 116, 0]);
        sys.write(f1, "This is a test.".as_bytes()).unwrap();
        let mut buffer = [0; 50];
        sys.close(f1).unwrap();
        sys.assert_block(8, 0, &[84, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101, 115, 116, 46]);
        sys.assert_block(2, 0, &[16, 0, 7]);
        sys.assert_block(2, 10, &[15, 0, 8]);
        let f2 = sys.open_read("one.txt").unwrap();
        let bytes_read = sys.read(f2, &mut buffer).unwrap();
        assert_eq!(bytes_read, 15);
        let s = core::str::from_utf8(&buffer[0..bytes_read]).unwrap();
        assert_eq!(s, "This is a test.");
    }

    const LONG_DATA: &str = "This is a much, much longer message.
    It crosses a number of different lines in the text editor, all synthesized
    with the goal of exceeding the 64 byte block limit by a considerable amount.
    To that end, this text contains considerable excessive verbiage.";

    #[test]
    fn test_long_write() {
        assert_eq!(265, LONG_DATA.len());
        let mut sys = make_small_fs();
        let f1 = sys.open_create("one.txt").unwrap();
        sys.write(f1, LONG_DATA.as_bytes()).unwrap();
        sys.close(f1);
        sys.assert_block(0, 0, &[3, 0, 0]);
        sys.assert_block(1, 0, &[255, 31, 0]);
        sys.assert_block(2, 0, &[16, 0, 7]);
        sys.assert_block(2, 10, &[9, 1, 8, 9, 10, 11, 12]);
        let read = read_to_string(&mut sys, "one.txt");
        assert_eq!(read.as_str(), LONG_DATA);
    }

    fn read_to_string(
        sys: &mut FileSystem<16, BLOCK_SIZE, 255, 8, 512, 32, 8>,
        filename: &str,
    ) -> String {

        let fd = sys.open_read(filename).unwrap();

        let mut read = String::new();
        let mut buffer = [0; 10];
        loop {
            let num_bytes = sys.read(fd, &mut buffer).unwrap();

            let s = core::str::from_utf8(&buffer[0..num_bytes]).unwrap();
            read.push_str(s);
            if num_bytes < buffer.len() {
                sys.close(fd).unwrap();
                return read;
            }
        }
    }

    #[test]
    fn test_complex_1() {
        let one = "This is a message, a short message, but an increasingly long message.
        This is a message, a short message, but an increasingly long message.";
        let two = "This is the second message I have chosen to undertake in this particular test.
        This is a continuation of this ever-so-controversial second message.\n";
        let mut sys = make_small_fs();
        let f1 = sys.open_create("one.txt").unwrap();

        sys.write(f1, one[0..one.len() / 2].as_bytes()).unwrap();
        let f2 = sys.open_create("two.txt").unwrap();

        sys.write(f2, two[0..two.len() / 2].as_bytes()).unwrap();
        sys.write(f1, one[one.len() / 2..one.len()].as_bytes())
            .unwrap();
        sys.write(f2, two[two.len() / 2..two.len()].as_bytes())
            .unwrap();
        sys.close(f1).unwrap();
        sys.close(f2).unwrap();
        assert_eq!(one, read_to_string(&mut sys, "one.txt").as_str());
        assert_eq!(two, read_to_string(&mut sys, "two.txt").as_str());
    }

    #[test]
    fn test_complex_2() {
        let one = "This is a message, a short message, but an increasingly long message.
        This is a message, a short message, but an increasingly long message.";
        let two = "This is the second message I have chosen to undertake in this particular test.
        This is a continuation of this ever-so-controversial second message.\n";
        let mut sys = make_small_fs();
        let f1 = sys.open_create("one.txt").unwrap();
        sys.write(f1, one[0..one.len() / 2].as_bytes()).unwrap();
        let f2 = sys.open_create("two.txt").unwrap();
        sys.write(f2, two[0..two.len() / 2].as_bytes()).unwrap();
        sys.close(f1).unwrap();
        sys.close(f2).unwrap();

        let f3 = sys.open_append("two.txt").unwrap();
        let f4 = sys.open_append("one.txt").unwrap();
        sys.write(f4, one[one.len() / 2..one.len()].as_bytes())
            .unwrap();
        sys.write(f3, two[two.len() / 2..two.len()].as_bytes())
            .unwrap();
        sys.close(f1).unwrap();
        sys.close(f2).unwrap();
        assert_eq!(one, read_to_string(&mut sys, "one.txt").as_str());
        assert_eq!(two, read_to_string(&mut sys, "two.txt").as_str());
    }

    #[test]
    fn test_complex_3() {
        let one = "This is a message, a short message, but an increasingly long message.
        This is a message, a short message, but an increasingly long message.";
        let two = "This is the second message I have chosen to undertake in this particular test.
        This is a continuation of this ever-so-controversial second message.\n";
        let mut sys = make_small_fs();
        let f1 = sys.open_create("one.txt").unwrap();
        sys.write(f1, one.as_bytes()).unwrap();
        sys.close(f1).unwrap();

        let f2 = sys.open_create("one.txt").unwrap();
        sys.write(f2, two.as_bytes()).unwrap();
        sys.close(f2).unwrap();

        assert_eq!(two, read_to_string(&mut sys, "one.txt").as_str());
    }

    #[test]
    fn test_file_not_found() {
        let mut sys = make_small_fs();
        let f1 = sys.open_create("one.txt").unwrap();
        sys.write(f1, "This is a test.".as_bytes()).unwrap();
        sys.close(f1).unwrap();
        match sys.open_read("one.tx") {
            FileSystemResult::Ok(_) => panic!("Shouldn't have found the file"),
            FileSystemResult::Err(e) => assert_eq!(e, FileSystemError::FileNotFound),
        }
    }

    #[test]
    fn test_file_not_open() {
        let mut sys = make_small_fs();
        let f1 = sys.open_create("one.txt").unwrap();
        sys.write(f1, "This is a test.".as_bytes()).unwrap();
        sys.close(f1).unwrap();
        let fd = sys.open_read("one.txt").unwrap();
        let mut buffer = [0; 10];
        match sys.read(fd + 1, &mut buffer) {
            FileSystemResult::Ok(_) => panic!("Should be an error!"),
            FileSystemResult::Err(e) => assert_eq!(e, FileSystemError::FileNotOpen),
        }
    }

    #[test]
    fn test_not_open_for_read() {
        let mut sys = make_small_fs();
        let f1 = sys.open_create("one.txt").unwrap();
        sys.write(f1, "This is a test.".as_bytes()).unwrap();
        let mut buffer = [0; 10];
        match sys.read(f1, &mut buffer) {
            FileSystemResult::Ok(_) => panic!("Should not work!"),
            FileSystemResult::Err(e) => assert_eq!(e, FileSystemError::NotOpenForRead),
        }
    }

    #[test]
    fn test_not_open_for_write() {
        let mut sys = make_small_fs();
        let f1 = sys.open_create("one.txt").unwrap();
        sys.write(f1, "This is a test.".as_bytes()).unwrap();
        sys.close(f1).unwrap();
        let f2 = sys.open_read("one.txt").unwrap();
        match sys.write(f2, "this is also a test".as_bytes()) {
            FileSystemResult::Ok(_) => panic!("Should be an error"),
            FileSystemResult::Err(e) => assert_eq!(e, FileSystemError::NotOpenForWrite),
        }
    }

    #[test]
    fn test_filename_too_long() {
        let mut sys = make_small_fs();
        match sys.open_create("this_is_an_exceedingly_long_filename_to_use.txt") {
            FileSystemResult::Ok(_) => panic!("This should be an error"),
            FileSystemResult::Err(e) => assert_eq!(e, FileSystemError::FilenameTooLong),
        }
    }

    #[test]
    fn test_already_open() {
        let mut sys = make_small_fs();
        let f1 = sys.open_create("one.txt").unwrap();
        sys.write(f1, "This is a test.".as_bytes()).unwrap();
        match sys.open_read("one.txt") {
            FileSystemResult::Ok(_) => panic!("Should be an error"),
            FileSystemResult::Err(e) => assert_eq!(e, FileSystemError::AlreadyOpen),
        }
    }

    #[test]
    fn test_file_too_big() {
        let mut sys = make_small_fs();
        let f1 = sys.open_create("one.txt").unwrap();
        for _ in 0..sys.max_file_size() - 1 {
            sys.write(f1, "A".as_bytes()).unwrap();
        }
        match sys.write(f1, "B".as_bytes()) {
            FileSystemResult::Ok(_) => panic!("Should be an error!"),
            FileSystemResult::Err(e) => assert_eq!(e, FileSystemError::FileTooBig),
        }
    }

    #[test]
    fn test_too_many_files() {
        let mut sys = make_small_fs();
        for i in 0..MAX_FILES_STORED - 1 {
            let filename = format!("file{i}");
            let f = sys.open_create(filename.as_str()).unwrap();
            let content = format!("This is sentence {i}");
            sys.write(f, content.as_bytes()).unwrap();
            sys.close(f).unwrap();
        }
        match sys.open_create("Final") {
            FileSystemResult::Ok(_) => panic!("This should be an error!"),
            FileSystemResult::Err(e) => assert_eq!(e, FileSystemError::TooManyFiles),
        }
    }

    // Disregard this test - too many valid possible solutions will fail it.
    #[test]
    fn test_disk_full() {
        let mut sys = make_small_fs();
        for i in 0..MAX_FILES_STORED - 1 {
            let filename = format!("file{i}");
            let f = sys.open_create(filename.as_str()).unwrap();
            for j in 0..sys.max_file_size() - 1 {

                match sys.write(f, "A".as_bytes()) {
                    FileSystemResult::Ok(_) => {}
                    FileSystemResult::Err(e) => {
                        assert_eq!(i, 30);
                        assert_eq!(j, 191);
                        assert_eq!(e, FileSystemError::DiskFull);
                        return;
                    }
                }
            }
            sys.close(f).unwrap();
        }
        panic!("The disk should have been full!");
    }
}